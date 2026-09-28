import numpy as np
import importlib
import torch
import torch.nn.functional as F
from . import tools as utils
import math

try:
    inference = importlib.import_module("inference")
except ImportError:
    inference = None

try:
    _metrics = importlib.import_module("evaluation.metrics")
    get_metrics = _metrics.get_metrics
    get_metrics_restr = _metrics.get_metrics_restr
except ImportError:
    def get_metrics(*args, **kwargs):
        raise ImportError("evaluation.metrics is not available in this workspace")

    def get_metrics_restr(*args, **kwargs):
        raise ImportError("evaluation.metrics is not available in this workspace")


class Evaluator:

    def __init__(self, batch_size=32, device='cuda', metrics='restr', strategy='overlapping'):

        self.batch_size = batch_size
        self.device = device

        if metrics == 'all':
            self.metrics = ["AUC-PR", "AUC-ROC",
                            "VUS-PR", "VUS-ROC", "Standard-F1"]
            self.metrics_fnc = get_metrics
        elif metrics == 'restr':
            self.metrics = ["AUC-PR", "AUC-ROC", "Standard-F1"]
            self.metrics_fnc = get_metrics_restr
        else:
            raise ValueError(f"Unknown metrics setting: {metrics}")

        self.strategy = strategy
        # if strategy == "overlapping":
        #     self.inference_strategy = inference.combined_pointwise_profile
        # elif strategy == "disjoint":
        #     self.inference_strategy = inference.disjoint_pointwise_profile
        # elif strategy == "MSE":
        #     self.inference_strategy = inference.rowwise_mse
        # else:
        #     raise ValueError(
                # f"Unknown inference strategy: {strategy}")

    def evaluate(self, data, label, model, win_size, stride=1):

        reconstruction_error = self.reconstruction_error(data, model, win_size, stride)

        rank = utils.find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        result = self.metrics_fnc(
            reconstruction_error, label, slidingWindow=rank)

        return result

    def reconstruction_error(self, data, model, win_size, stride=1, mask_ratio=0.3):


        data = data.astype(np.float32)

        model = model.to(self.device)
        model.eval()

        data = data.astype(np.float32)

        if self.strategy == "overlapping":
            reconstruction_error = self._overlapping_reconstruction(
                data, model, win_size, stride)
        elif self.strategy == "disjoint":
            reconstruction_error = self._disjoint_reconstruction(
                data, model, win_size)
        elif self.strategy == "MSE":
            reconstruction_error = self._mse_reconstruction(
                data, model, win_size)
        elif self.strategy == "masked_sliding":
            reconstruction_error = self._masked_sliding_reconstruction(
                data, model, win_size, mask_ratio=mask_ratio) 
        else:
            raise ValueError(
                f"Unknown inference strategy: {self.strategy}")


        return reconstruction_error.cpu().numpy()

    def relative_reconstruction_error(self, data, model, win_size, stride=1):
        if self.strategy == "overlapping":
            n = data.shape[0]
            sum_err = torch.zeros(n, device=self.device, dtype=torch.float32)
            mse_fn = torch.nn.L1Loss(reduction="none")
            start = 0

            ds = utils.ReconstructDataset(
            data, window_size=win_size, stride=stride, normalize=False)
            dl = torch.utils.data.DataLoader(
                ds, batch_size=self.batch_size, shuffle=False, drop_last=False, )

            with torch.inference_mode():
                for xb in dl:
                    xb = xb.to(self.device).float()
                    out = model(xb)

                    error = mse_fn(out, xb).sum(dim=-1)  # (B,W)
                    sum_err = self.update_sum_with_batch_fold(
                        sum_err, error, start, n, win_size, stride)
                    start += xb.shape[0] * stride
            rec_err = self.finalize_avg(
                sum_err, self._count_closed_form(n, win_size, stride, self.device))
            
        elif self.strategy == "disjoint":
            n = data.shape[0]
            rec_err = torch.zeros(n, device=self.device, dtype=torch.float32)
            mse_fn = torch.nn.MSELoss(reduction="none")
            start = 0

            ds = utils.ReconstructDataset(
            data, window_size=win_size, stride=win_size, normalize=False, add_last_partial=True)
            # since stride > 1, the last window is partial and is stored as data[-window_size:]
            dl = torch.utils.data.DataLoader(
                ds, batch_size=self.batch_size, shuffle=False, drop_last=False, )

            with torch.inference_mode():
                for xb in dl:
                    xb = xb.to(self.device).float()
                    out = model(xb)

                    error = mse_fn(out, xb)  # (B,W)
                    end = start + xb.shape[0] * win_size

                    if end > n:
                        end = n
                        full_windows = (xb.shape[0]-1) * win_size 
                        remainder = n % win_size
                        if remainder == 0:
                            remainder = win_size
                        rec_err[start:start+full_windows] = error.flatten()[:full_windows]
                        rec_err[-remainder:] = error[-1, -remainder:].flatten()
                        break
                    else:
                        rec_err[start:end] = error.flatten()
                        start = end

        return rec_err


    @torch.no_grad()
    def _count_closed_form(self, n: int, w: int, s: int, device=None) -> torch.Tensor:
        i = torch.arange(n, device=device)
        lo = (i - w + 1).clamp_min(0)
        hi = i.clamp_max(n - w)
        cnt = (hi // s) - ((lo + s - 1) // s) + 1
        return cnt.clamp_min(0).to(torch.float32)  # (n,)

    @torch.no_grad()
    def init_state(self, n: int, w: int, s: int, device=None, dtype=torch.float32):
        sum_err = torch.zeros(n, device=device, dtype=dtype)
        count = self._count_closed_form(
            n, w, s, device=device)  # build once, cache
        return sum_err, count

    @torch.no_grad()
    def update_sum_with_batch_fold(self, sum_err: torch.Tensor, error_batch: torch.Tensor, start: int, n: int, win_size: int, stride: int):
        """
        sum_err: (n,)
        E_batch: (B, w) pointwise errors for windows k=start..start+b-1 (contiguous!)
        k0: global window index of the first window in this batch
        """
        b = error_batch.shape[0]
        if b == 0:
            return sum_err

        # fold expects (N, C*prod(kernel), L) -> here (1, w, B)
        cols = error_batch.transpose(0, 1).unsqueeze(0)  # (1, w, B)

        # length covered by this batch in time axis
        seg_len = (b - 1) * stride + win_size
        seg = F.fold(cols, output_size=(1, seg_len), kernel_size=(
            1, win_size), stride=(1, stride)).view(-1)  # (seg_len,)

        end = min(start + seg_len, n)                    # clip for safety
        sum_err[start:end] += seg[: (end - start)]

        return sum_err

    @torch.no_grad()
    def finalize_avg(self, sum_err: torch.Tensor, count: torch.Tensor):
        return sum_err / count.clamp_min(1)

    @torch.no_grad()
    def _overlapping_reconstruction(self, data, model, win_size, stride):
        n = data.shape[0]
        sum_err = torch.zeros(n, device=self.device, dtype=torch.float32)
        mse_fn = torch.nn.MSELoss(reduction="none")
        start = 0

        ds = utils.ReconstructDataset(
        data, window_size=win_size, stride=stride, normalize=False)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, drop_last=False, )

        with torch.inference_mode():
            for xb in dl:
                xb = xb.to(self.device).float()
                out = model(xb)

                error = mse_fn(out, xb).sum(dim=-1)  # (B,W)
                sum_err = self.update_sum_with_batch_fold(
                    sum_err, error, start, n, win_size, stride)
                start += xb.shape[0] * stride
        rec_err = self.finalize_avg(
            sum_err, self._count_closed_form(n, win_size, stride, self.device))
        return rec_err
    
    @torch.no_grad()
    def overlapping_reconstruct(self, data, model, win_size, stride):
        n = data.shape[0]
        recon = torch.zeros(n, device=self.device, dtype=torch.float32)
        start = 0

        ds = utils.ReconstructDataset(
        data, window_size=win_size, stride=stride, normalize=False)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, drop_last=False, )

        with torch.inference_mode():
            for xb in dl:
                xb = xb.to(self.device).float()
                out = model(xb).sum(dim=-1)  # (B,)

                recon = self.update_sum_with_batch_fold(
                    recon, out, start, n, win_size, stride)
                start += xb.shape[0] * stride
        rec_err = self.finalize_avg(
            recon, self._count_closed_form(n, win_size, stride, self.device))
        return rec_err
    

    @torch.no_grad()
    def _disjoint_reconstruction(self, data, model, win_size):
        n = data.shape[0]
        rec_err = torch.zeros(n, device=self.device, dtype=torch.float32)
        mse_fn = torch.nn.MSELoss(reduction="none")
        start = 0

        ds = utils.ReconstructDataset(
        data, window_size=win_size, stride=win_size, normalize=False, add_last_partial=True)
        # since stride > 1, the last window is partial and is stored as data[-window_size:]
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, drop_last=False, )

        with torch.inference_mode():
            for xb in dl:
                xb = xb.to(self.device).float()
                out = model(xb)

                error = mse_fn(out, xb)  # (B,W)
                end = start + xb.shape[0] * win_size

                if end > n:
                    end = n
                    full_windows = (xb.shape[0]-1) * win_size 
                    remainder = n % win_size
                    if remainder == 0:
                        remainder = win_size
                    rec_err[start:start+full_windows] = error.flatten()[:full_windows]
                    rec_err[-remainder:] = error[-1, -remainder:].flatten()
                    break
                else:
                    rec_err[start:end] = error.flatten()
                    start = end


        return rec_err
    
    @torch.no_grad()
    def _mse_reconstruction(self, data, model, win_size):
        n = data.shape[0]
        pad_left = math.ceil((win_size - 1) / 2)
        pad_right = (win_size - 1) // 2
        rec_err = torch.zeros(n, device=self.device, dtype=torch.float32)
        mse_fn = torch.nn.MSELoss(reduction="none")
        start = pad_left

        ds = utils.ReconstructDataset(
        data, window_size=win_size, stride=1, normalize=False)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, drop_last=False, )
        with torch.inference_mode():
            for xb in dl:
                xb = xb.to(self.device).float()
                out = model(xb)

                error = mse_fn(out, xb).mean(dim=1)  # (B,W)

                end = start + xb.shape[0]
                rec_err[start:end] = error.flatten()
                start = end

     
        
        rec_err[:pad_left] = rec_err[pad_left]
        rec_err[n - pad_right:] = rec_err[n - pad_right - 1]

        return rec_err

    @torch.no_grad()
    def _masked_sliding_reconstruction(self, data, model, win_size, mask_ratio=0.15):
        n = data.shape[0]

        rec_err = torch.zeros(n, device=self.device, dtype=torch.float32)
        count = torch.zeros(n, device=self.device, dtype=torch.float32)
        mse_fn = torch.nn.MSELoss(reduction="none")

        ds = utils.ReconstructDataset(data, window_size=win_size, stride=1, normalize=False)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # mask the central section of each window
        len_mask = max(1, int(win_size * mask_ratio))  # keep at least 1
        center_start = (win_size - len_mask) // 2
        center_end = center_start + len_mask

        # `start` = window start index in the original series (since stride=1)
        start = 0

        with torch.inference_mode():
            for xb in dl:
                xb = xb.to(self.device).float()  # (B, win_size) expected

                xb_masked = xb.clone()
                xb_masked[:, center_start:center_end] = 0.0  # mask center

                out = model(xb_masked)
                error = mse_fn(out, xb)  # (B, win_size)

                B = xb.shape[0]

                # For window k starting at original index (start + k),
                # masked region maps to original indices:
                # (start + k + center_start) .. (start + k + center_end - 1)
                base = torch.arange(B, device=self.device) + start  # window starts in original series
                idx = base[:, None] + torch.arange(center_start, center_end, device=self.device)[None, :]

                # write masked-region errors into rec_err
                valid = (idx >= 0) & (idx < n)
                error_slice = error[:, center_start:center_end]  # (B, len_mask)

                # flatten everything to 1D before boolean indexing / assignment
                idx_flat = idx.reshape(-1)                       # (B*len_mask,)
                err_flat = error_slice.reshape(-1)               # (B*len_mask,)
                valid_flat = valid.reshape(-1)                   # (B*len_mask,)

                rec_err[idx_flat[valid_flat]] += err_flat[valid_flat]
                count[idx_flat[valid_flat]] += 1.0

                start += B  # stride=1 -> next batch windows start B later

        # fill edges that cannot be “center-covered” by a full window
        rec_err[:center_start] = rec_err[center_start]
        rec_err[n - center_start:] = rec_err[n - center_start - 1]
        count[:center_start] = count[center_start]
        count[n - center_start:] = count[n - center_start - 1]
        rec_err = rec_err / count.clamp_min(1)

        return rec_err

    @torch.no_grad()
    def reconstruct(self, data, model, win_size):
        model = model.to(self.device)
        model.eval()
        data = data.astype(np.float32)
        n = data.shape[0]
        ds = utils.ReconstructDataset(
            data, window_size=win_size, stride=1, normalize=False)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, drop_last=False, )
        outs = []
        with torch.inference_mode():
            for xb in dl:
                xb = xb.to(self.device).float()
                out = model(xb)

                outs.append(out.cpu())

        output = torch.cat(outs, dim=0).numpy()[:, :, 0]
        return output

    def reconstruction_error_anomaly_transformer(self, data, model, win_size):

        temperature = 50  #  same value as in the original github repo
        model = model.to(self.device)
        model.eval()

        data = data.astype(np.float32)
        n = data.shape[0]

        ds = utils.ReconstructDataset(
            data, window_size=win_size, stride=1, normalize=False
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        criterion = torch.nn.MSELoss(reduction="none")
        attens_energy = []
        with torch.inference_mode():
            for batch in dl:
                input = batch.to(self.device)
                output, series, prior, _ = model(input)

                recon_loss = torch.mean(criterion(input, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = utils.my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   win_size)).detach()) * temperature
                        prior_loss = utils.my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += utils.my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   win_size)).detach()) * temperature
                        prior_loss += utils.my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    win_size)),
                            series[u].detach()) * temperature
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)

                cri = metric * recon_loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)

        # since we use overlapping windows, we need to aggregate the pointwise energy
        # reshape to (num_windows, window_size)
        test_energy = test_energy.reshape(-1, win_size)
        reconstruction_error = self.inference_strategy(
            test_energy, n, win_size)

        return reconstruction_error
