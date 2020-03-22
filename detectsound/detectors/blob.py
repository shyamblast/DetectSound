
import abc
import numpy as np
from scipy.signal import convolve, sosfiltfilt
from scipy.optimize import linear_sum_assignment

from detectsound.utils import get_1d_gauss_kernel, get_1d_LoG_kernel, first_true, last_true


class _BlobExtractor:
    """
    A general-purpose detector for extracting blob-like features from
    spectrogram-like inputs. It is an implementation of the algorithm described
    in -
      Shyam Madhusudhana, Alexander Gavrilov, and Christine Erbe. "A general
      purpose automatic detector of broadband transient signals in underwater
      audio." In 2018 OCEANS-MTS/IEEE Kobe Techno-Oceans (OTO), pp. 1-6.
      IEEE, 2018.
    This class defines the base operations that will be used in both batch-mode
    and streaming-mode applications.

    Parameters
    ----------
    num_f_bins : int
        The height of spectrogram-like 2d inputs.
    centroid_bounds : 2 element list or tuple of ints
        Defines the range of bins (y-axis indices) in inputs within which to
        look for 1d plateaus.
    min_snr : float
        A threshold for discarding low SNR 1d plateaus in each frame.
    min_frames : int
        The minimum number of frames that a traced blob must span to be
        considered valid.
    cost_std : 3 element list or tuple of floats
        Standard deviation of the vector [center bin, height, snr] (as
        applies to the leading end of a blob being traced) for use in
        computing Mahalanobis distances as costs for associating candidate
        extensions to blobs being traced.
    first_scale : int or float
        Sigma value of the first scale Laplacian of Gaussian operator. The
        remaining scales will be automatically populated as a geometric
        progression (first_scale * 2^n) to fit within num_f_bins.
    t_blur_sigma : int or float, optional
        If not None, the value will define a Gaussian kernel that will be
        convolved with the inputs along the time axis to smoothen highly rapid
        fluctuations.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, num_f_bins, centroid_bounds, min_snr,
                 min_frames, cost_std,
                 first_scale=2., t_blur_sigma=None):

        self._threshold = min_snr
        self._min_width = min_frames
        self._centroid_bounds = centroid_bounds

        self._cost_std = np.asarray(cost_std, dtype=np.float32)

        self._t_conv_sigma = t_blur_sigma
        if t_blur_sigma is not None:
            kernel = get_1d_gauss_kernel(self._t_conv_sigma)
            self._t_conv_kernel = np.reshape(kernel, [1, int(len(kernel))])
        else:
            self._t_conv_kernel = None

        # Determine num scales possible and populate sigmas accordingly
        n = np.arange(
            np.floor(np.log2((num_f_bins - 1) / ((2 * 3) * first_scale)) + 1),
            dtype=np.int)
        self._f_conv_sigmas = first_scale * (2 ** n)

        self._f_conv_kernels = [get_1d_LoG_kernel(sigma)
                                for sigma in self._f_conv_sigmas]
        self._f_conv_kernels = [np.reshape(kernel, [len(kernel), 1])
                                for kernel in self._f_conv_kernels]

    @abc.abstractmethod
    def extract_blobs(self, frames):
        pass

    def _process_frames(self, padded_frames, active_blobs, frame_offset=0):
        """The workhorse of the class. Works with both batch- and streaming-mode
        applications.

        - Input spectrogram frames must be pre-padded. If batch mode, pad with 0
          on both left & right for N frames. If streaming mode, concatenate only
          left with previous N frames. N depends on t_conv_sigma.
        - 'active_blobs' will be written to. Pass an empty list variable if
          batch mode, else a list containing "active" blobs traced up to this
          invocation.
        - 'frame_offset' is only to be used in streaming mode to adjust for the
          starting frame's index.

        Returns a list of any blobs that ended before the last input frame.
        """

        DISALLOWED = float(1e30)

        # Temporal blurring, with a Gaussian kernel along x-axis only:
        # Get only the valid points after convolution
        frames = padded_frames if self._t_conv_sigma is None else \
            convolve(padded_frames, self._t_conv_kernel, mode='valid')

        num_f, num_t = frames.shape[0], frames.shape[1]

        scale_space = self._generate_scale_space(frames)

        salient_pts_mask, zc_or_valley_mask = \
            _BlobExtractor._get_zero_crossing_and_salient_scale_masks(
                scale_space)

        # Throw away the smallest scale elements.
        # Note: it's already been done so for salient_pts_mask and
        # zc_or_valley_mask.
        scale_space = scale_space[:, :, 1:]

        # To hold blobs that were done tracing before the current frame
        inactive_blobs = list()

        # Process frames iteratively
        for frame_idx in range(num_t):

            # "cand"idate extensions
            cand_centers_f, cand_scales = \
                np.where(salient_pts_mask[
                    self._centroid_bounds[0]:self._centroid_bounds[1] + 1,
                    frame_idx, :])

            # Adjust for clipped input
            cand_centers_f += self._centroid_bounds[0]

            # Gather candidate info (F edges & SNRs)
            num_cands = len(cand_centers_f)
            if num_cands > 0:

                # 2d array of F extents for each candidate
                cand_edges = np.asarray([
                    [
                        last_true(
                            zc_or_valley_mask[:center_f, frame_idx, scale],
                            0),
                        first_true(
                            zc_or_valley_mask[(center_f + 1):,
                                              frame_idx,
                                              scale],
                            num_f - 2 - center_f) + center_f + 1]
                    for center_f, scale in zip(cand_centers_f, cand_scales)],
                    dtype=np.uint32)

                # Candidates' SNRs (height in scale space)
                cand_snrs = np.asarray([
                    scale_space[f_idx, frame_idx, s_idx]
                    for f_idx, s_idx in zip(cand_centers_f, cand_scales)])

            else:
                cand_edges = np.zeros((0, 2), dtype=np.uint32)
                cand_snrs = np.zeros((0,))

            # Initialize mask
            unused_cands_mask = np.full((num_cands,), True, dtype=np.bool)

            num_active_blobs = len(active_blobs)
            if num_active_blobs > 0:

                # Determine "costs" of assigning a candidate to an active blob
                costs = np.stack([
                    blob.validate_and_measure_costs(
                        cand_centers_f, cand_edges, cand_snrs,
                        self._cost_std, DISALLOWED)
                    for blob in active_blobs])

                # Mark very-high cost assignments as DISALLOWED
                # costs[costs > (3. ** 2) * 3] = DISALLOWED

                # Solve the least-cost assignment problem
                blob_idxs, cand_idxs = linear_sum_assignment(costs)

                # Only retain valid pairings
                temp = np.asarray(
                    [costs[blob_idx, cand_idx] < DISALLOWED
                     for blob_idx, cand_idx in zip(blob_idxs, cand_idxs)],
                    dtype=np.bool)      # Get the mask
                blob_idxs, cand_idxs = blob_idxs[temp], cand_idxs[temp]

                # Blobs with a matched extension candidate
                for blob_idx, cand_idx in zip(blob_idxs, cand_idxs):
                    active_blobs[blob_idx].extend(cand_centers_f[cand_idx],
                                                  cand_edges[cand_idx, :],
                                                  cand_snrs[cand_idx])

                # Mark unused candidates for possibly starting new "active"
                # blobs
                unused_cands_mask[cand_idxs] = False

                # Move blobs without matched extensions into "inactive" list if
                # they are long enough
                unextendable_blobs_mask = np.full((num_active_blobs,), True,
                                                  dtype=np.bool)
                unextendable_blobs_mask[blob_idxs] = False
                for blob_idx in np.flip(np.where(unextendable_blobs_mask)[0]):
                    # Popping in reverse order so that indices remain intact
                    finished_blob = active_blobs.pop(blob_idx)
                    if finished_blob.width >= self._min_width:
                        inactive_blobs.append(finished_blob.finalize())

            # Unassigned candidates. Start new "active" blobs from them if they
            # satisfy threshold criterion.
            for bin_idx, edge_idxs, snr in \
                zip(cand_centers_f[unused_cands_mask],
                    cand_edges[unused_cands_mask, :],
                    cand_snrs[unused_cands_mask]):
                if snr >= self._threshold:
                    active_blobs.append(_Blob(frame_offset + frame_idx,
                                              bin_idx, edge_idxs, snr))

        return inactive_blobs

    def _generate_scale_space(self, surface):
        """Apply LoG filters at chosen scales after padding 'surface'
        appropriately."""

        num_f, num_t = surface.shape[0], surface.shape[1]

        # Preallocate
        scale_space = np.zeros((num_f, num_t, len(self._f_conv_sigmas)),
                               dtype=np.float32)

        # Process at all scales
        prev_scale_padding = 0
        in_surface = surface
        for scale_idx, scale_kernel in enumerate(self._f_conv_kernels):

            # Add padding (incrementally) prior to convolutions so that values
            # at boundaries are not very unrealistic.
            curr_scale_padding = len(scale_kernel) // 2
            incr_padding = curr_scale_padding - prev_scale_padding
            in_surface = np.pad(in_surface,
                                [[incr_padding, incr_padding], [0, 0]],
                                'symmetric')

            # Apply LoG filter
            scale_space[:, :, scale_idx] = \
                convolve(in_surface, scale_kernel, mode='valid')

            # Update for next iteration
            prev_scale_padding = curr_scale_padding

        return scale_space

    @staticmethod
    def _get_zero_crossing_and_salient_scale_masks(scale_space, min_height=0):

        # TODO: the 'saliency' search can be restricted to the "valid"
        # user-chosen bandwidth. If done, zero crossing (and valley) masks must
        # still be done for the whole frequency range.

        # Nomenclature guide for below 4 operations:
        #  gt = greater than
        #  nf = next frequency bin, pf = previous frequency bin
        #  ns = next scale
        gt_nf = np.greater(
            np.pad(scale_space, ((1, 0), (0, 0), (0, 0)), 'constant',
                   constant_values=-np.inf),
            np.pad(scale_space, ((0, 1), (0, 0), (0, 0)), 'constant',
                   constant_values=-np.inf))
        gt_ns = np.greater(
            np.pad(scale_space, ((0, 0), (0, 0), (1, 0)), 'constant',
                   constant_values=-np.inf),
            np.pad(scale_space, ((0, 0), (0, 0), (0, 1)), 'constant',
                   constant_values=-np.inf))
        gt_nf_ns = np.greater(
            np.pad(scale_space, ((1, 0), (0, 0), (1, 0)), 'constant',
                   constant_values=-np.inf),
            np.pad(scale_space, ((0, 1), (0, 0), (0, 1)), 'constant',
                   constant_values=-np.inf))
        gt_pf_ns = np.greater(
            np.pad(scale_space, ((0, 1), (0, 0), (1, 0)), 'constant',
                   constant_values=-np.inf),
            np.pad(scale_space, ((1, 0), (0, 0), (0, 1)), 'constant',
                   constant_values=-np.inf))

        saliency_mask = np.all(
            np.stack([
                scale_space >= min_height,
                gt_nf[1:, :, :], np.logical_not(gt_nf[:-1, :, :]),
                gt_ns[:, :, 1:], np.logical_not(gt_ns[:, :, :-1]),
                gt_nf_ns[1:, :, 1:], np.logical_not(gt_nf_ns[:-1, :, :-1]),
                gt_pf_ns[:-1, :, 1:], np.logical_not(gt_pf_ns[1:, :, :-1])],
                axis=3),
            axis=3)

        scale_space_signs = \
            (np.sign(scale_space[:, :, 1:])).astype(dtype=np.int8)
        temp = np.abs(scale_space[:, :, 1:])
        lower_abs_mask = np.less(temp[:-1, :, :], temp[1:, :, :])

        # Find zero-crossings or valley points along frequency axis
        zcs = np.not_equal(scale_space_signs[:-1, :, :],
                           scale_space_signs[1:, :, :])
        zcs_or_valleys_mask = np.any(np.stack([
            np.pad(np.logical_and(zcs, lower_abs_mask),
                   ((0, 1), (0, 0), (0, 0)), 'constant', constant_values=True),
            np.pad(np.logical_and(zcs, np.logical_not(lower_abs_mask)),
                   ((1, 0), (0, 0), (0, 0)), 'constant', constant_values=True),
            np.logical_and(np.logical_not(gt_nf[1:, :, 1:]), gt_nf[:-1, :, 1:])
            ], axis=3), axis=3)

        # Throw away the smallest scale elements. Note: for
        # zcs_or_valleys_mask, the discarding was already done.
        return saliency_mask[:, :, 1:], zcs_or_valleys_mask


class BlobExtractor(_BlobExtractor):
    """
    A general-purpose detector for extracting blob-like features from
    spectrogram-like inputs. It is an implementation of the algorithm described
    in -
      Shyam Madhusudhana, Alexander Gavrilov, and Christine Erbe. "A general
      purpose automatic detector of broadband transient signals in underwater
      audio." In 2018 OCEANS-MTS/IEEE Kobe Techno-Oceans (OTO), pp. 1-6.
      IEEE, 2018.

    Parameters
    ----------
    num_f_bins : int
        The height of spectrogram-like 2d inputs.
    centroid_bounds : 2 element list or tuple of ints
        Defines the range of bins (y-axis indices) in inputs within which to
        look for 1d plateaus.
    min_snr : float
        A threshold for discarding low SNR 1d plateaus in each frame.
    min_frames : int
        The minimum number of frames that a traced blob must span to be
        considered valid.
    cost_std : 3 element list or tuple of floats
        Standard deviation of the vector [center bin, height, snr] (as
        applies to the leading end of a blob being traced) for use in
        computing Mahalanobis distances as costs for associating candidate
        extensions to blobs being traced.
    first_scale : int or float
        Sigma value of the first scale Laplacian of Gaussian operator. The
        remaining scales will be automatically populated as a geometric
        progression (first_scale * 2^n) to fit within num_f_bins.
    t_blur_sigma : int or float, optional
        If not None, the value will define a Gaussian kernel that will be
        convolved with the inputs along the time axis to smoothen highly rapid
        fluctuations.
    """

    def __init__(self, num_f_bins, centroid_bounds, min_snr,
                 min_frames, cost_std,
                 first_scale=2., t_blur_sigma=None):

        super(BlobExtractor, self).__init__(num_f_bins,
                                            centroid_bounds, min_snr,
                                            min_frames, cost_std,
                                            first_scale, t_blur_sigma)

    def extract_blobs(self, frames):
        """Extract all blobs from 'frames' that satisfy the criteria defined.
        'frames' must be a 2d array with its height equal 'num_f_bins' that was
        passed to the constructor.

        Returns a list of traced blobs."""

        # Add padding so that values at boundaries are not very unrealistic
        if self._t_conv_sigma is not None:
            t_blur_pad_len = int(np.ceil(3 * self._t_conv_sigma))
            padded_frames = np.pad(frames,
                                   [[0, 0], [t_blur_pad_len, t_blur_pad_len]],
                                   'symmetric')
        else:
            padded_frames = frames

        active_blobs = list()   # To hold blobs that haven't yet ended

        # Process frames
        completed_blobs = self._process_frames(padded_frames, active_blobs)

        # Move long-enough active blobs into the "completed" list
        for blob in active_blobs:
            if blob.width >= self._min_width:
                completed_blobs.append(blob.finalize())

        return completed_blobs


class _Blob:
    """An internal-use container class for managing blobs as they are traced
    iteratively across frames of a spectrogram."""

    def __init__(self, start_frame_idx,
                 start_bin_idx, start_edge_idxs, start_snr):

        # pre-allocate modest amount of space for storing data from
        # future frames
        self._bin_idxs = np.zeros((16,), dtype=np.uint16)
        self._edge_idxs = np.zeros((16, 2), dtype=np.uint16)
        self._snrs = np.zeros((16,), dtype=np.float32)

        self._start_frame_idx = start_frame_idx

        # (internal) index to the newest valid value in the above containers
        self._last_frame_idx = 0

        # copy the data about the first frame
        self._bin_idxs[0] = start_bin_idx
        self._edge_idxs[0, :] = start_edge_idxs
        self._snrs[0] = start_snr

    def extend(self, bin_idx, edge_idxs, snr):

        # expand storage if necessary
        if self._last_frame_idx == len(self._bin_idxs) - 1:
            self._bin_idxs = np.pad(self._bin_idxs, (0, 16),
                                    'constant', constant_values=0)
            self._edge_idxs = np.pad(self._edge_idxs, ((0, 16), (0, 0)),
                                     'constant', constant_values=0)
            self._snrs = np.pad(self._snrs, (0, 16),
                                'constant', constant_values=0)

        self._last_frame_idx += 1

        # copy the frame data
        self._bin_idxs[self._last_frame_idx] = bin_idx
        self._edge_idxs[self._last_frame_idx, :] = edge_idxs
        self._snrs[self._last_frame_idx] = snr

    def validate_and_measure_costs(self,
                                   candidate_centers,
                                   candidate_edges,
                                   candidate_snrs,
                                   distr_std, prohibited_cost=np.inf):
        """
        First, checks valid candidates based on whether ...
            i)  there is any overlap
            ii) candidate centroid is within blob's bounds.
        Then, for all valid candidates, computes the square of the Mahalanobis
        distances from the blob's frontier to each data point, using standard
        deviation 'distr_std'. Non-valid candidates are assigned
        'prohibited_cost'.
        Returns a 1d vector containing computed/prohibited costs.
        The returned values are generally only used in comparisons, without
        much meaning associated with the actual values. So, computing the final
        sqrt() computation of Mahalanobis distances is omitted.
        """

        end_bin_idx = self._bin_idxs[self._last_frame_idx]
        end_edge_idxs = self._edge_idxs[self._last_frame_idx]
        end_snr = self._snrs[self._last_frame_idx]

        # build a vector of [centroid, width, SNR]
        frontier = [float(end_bin_idx),
                    float(end_edge_idxs[1] - end_edge_idxs[0]),
                    end_snr]

        # data points
        data_points = np.stack([candidate_centers.astype(np.float32),
                                (candidate_edges[:, 1] -
                                 candidate_edges[:, 0]).astype(np.float32),
                                candidate_snrs]).T

        valid_cands_mask = np.all(
            np.stack([
                # Check for overlap
                (np.minimum(end_edge_idxs[1], candidate_edges[:, 1]) >
                 np.maximum(end_edge_idxs[0], candidate_edges[:, 0])),

                # Check if centroid is within blob's F extents
                end_edge_idxs[1] >= candidate_centers,
                end_edge_idxs[0] <= candidate_centers,

                # Other heuristics
                (np.abs(data_points[:, 0] - frontier[0]) / distr_std[0]) <= 4,
                (np.abs(data_points[:, 1] - frontier[1]) / distr_std[1]) <= 4,
                (np.abs(data_points[:, 2] - frontier[2]) / distr_std[2]) <= 4
            ]), axis=0)

        cost_to_candidate = np.full((len(candidate_centers),), prohibited_cost)
        cost_to_candidate[valid_cands_mask] = \
            (((np.expand_dims(frontier, 0) - data_points[valid_cands_mask]) /
              np.expand_dims(distr_std, 0)) ** 2).sum(axis=1)

        return cost_to_candidate

    def finalize(self):
        """To be called when tracing of an "active" blob ends."""

        # Discard any excess storage
        self._bin_idxs = self._bin_idxs[:self._last_frame_idx + 1]
        self._edge_idxs = self._edge_idxs[:self._last_frame_idx + 1, :]
        self._snrs = self._snrs[:self._last_frame_idx + 1]

        return self

    @property
    def first_frame(self):
        return self._start_frame_idx

    @property
    def last_frame(self):
        return self._start_frame_idx + self._last_frame_idx

    @property
    def width(self):
        return self._last_frame_idx + 1

    @property
    def bins(self):
        return self._bin_idxs[:self._last_frame_idx + 1]

    @property
    def edges(self):
        return self._edge_idxs[:self._last_frame_idx + 1, :]

    @property
    def snrs(self):
        return self._snrs[:self._last_frame_idx + 1]
