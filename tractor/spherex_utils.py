# This file is part of the `tractor-spherex` fork of the Tractor
# (https://github.com/Caltech-IPAC/tractor-spherex), created for the NASA SPHEREx
# mission's use by the SPHEREx Science Data Center team at Caltech/IPAC.
#
# Copyright (C) 2026, California Institute of Technology.
# Licensed under the GNU General Public License, version 2 (GPLv2),
# the same as the Tractor. See LICENSE/COPYING.
#
# This file provides PSF models specific to SPHEREx.

# Related third party imports
from astrometry.util.miscutils import get_overlapping_region
from cv2 import INTER_AREA, resize
import numpy as np

# Local application/library specific imports
from tractor.patch import Patch, ModelMask
from tractor.psf import lanczos_shift_image
from tractor.wcs import NullWCS

def getUnitFluxModelPatchSPHEREx(self, img, px=None, py=None, minval=0.0, modelMask=None,
                            force_halfsize=None, **kwargs):
    """Convolve the oversampled PSF with the galaxy shape model using
    Fourier transforms, then downsample the result onto the native
    resolution pixels.

    Notes
    -----
    1. The original method assumes the PSF is at image resolution, whereas
    the SPHEREx-specific method accounts for arbitrary PSF oversampling.
    2. The original method can render profile galaxies using a mixture
    of Gaussians or Fourier Transforms, whereas the SPHEREx-specific method
    only uses the latter.

    """
    if px is None or py is None:
        px, py = img.getWcs().positionToPixel(self.getPosition(), self)

    # Use arbitrary PSF oversampling
    psf = img.getPsf()
    oversample_raw = np.asarray(psf._psf.oversampling)
    if oversample_raw.size == 1:
        x_oversample = y_oversample = int(oversample_raw.item())
    elif oversample_raw.size == 2:
        y_oversample, x_oversample = map(int, oversample_raw)
    else:
        raise ValueError(
            f"Unexpected oversampling shape/value: "
            f"{oversample_raw.shape}, {oversample_raw}"
        )
    if x_oversample == y_oversample:
        oversample = x_oversample
    else:
        raise NotImplementedError("This path assumes equal x/y oversampling.")

    if hasattr(psf, 'getFourierTransformHiRes'):
        if modelMask is None:
            # choose the patch size
            halfsize = self._getUnitFluxPatchSize(img, px=px, py=py, minval=minval)
            # find overlapping pixels to render
            outx, inx = get_overlapping_region(
                int(np.floor(px - halfsize)), int(np.ceil(px + halfsize + 1)),
                0, img.getWidth())
            outy, iny = get_overlapping_region(
                int(np.floor(py - halfsize)), int(np.ceil(py + halfsize + 1)),
                0, img.getHeight())
            if inx == [] or iny == []:
                # no overlap
                return None
            x0, x1 = outx.start, outx.stop
            y0, y1 = outy.start, outy.stop
        else:
            x0, y0 = modelMask.x0, modelMask.y0

        # FFT in oversampled (larger) scale -------------------------------
        if modelMask is None:
            # Avoid huge galaxies
            # Let `halfsize` be smaller than or equal to the longer side of
            # image.
            imsz = max(img.shape)
            halfsize = min(halfsize, imsz)
        else:
            # modelMask sets the sizes
            mh, mw = modelMask.shape
            x1 = x0 + mw
            y1 = y0 + mw

            halfsize = max(mh / 2.0, mw / 2.0)
            # How far from the source center to furthest modelMask edge?
            # FIXME -- add 1 for Lanczos margin?
            halfsize = max(halfsize, max(max(1 + px - x0, 1 + x1 - px),
                                        max(1 + py - y0, 1 + y1 - py)))
            psfh, psfw = psf.shape
            halfsize = max(halfsize, max(psfw / 2.0, psfh / 2.0))
            if force_halfsize is not None:
                halfsize = force_halfsize

            # Is the source center outside the modelMask?
            sourceOut = (px < x0 or px > x1 - 1 or py < y0 or py > y1 - 1)

            if sourceOut:
                # FFT, modelMask, source is outside the box.
                neardx, neardy = 0., 0.
                if px < x0:
                    neardx = x0 - px
                if px > x1:
                    neardx = px - x1
                if py < y0:
                    neardy = y0 - py
                if py > y1:
                    neardy = py - y1
                nearest = np.hypot(neardx, neardy)
                if nearest > self.getRadius():
                    return None

                # How far is the furthest point from the source center?
                farw = max(abs(x0 - px), abs(x1 - px))
                farh = max(abs(y0 - py), abs(y1 - py))
                bigx0 = int(np.floor(px - farw))
                bigx1 = int(np.ceil(px + farw))
                bigy0 = int(np.floor(py - farh))
                bigy1 = int(np.ceil(py + farh))
                bigw = 1 + bigx1 - bigx0
                bigh = 1 + bigy1 - bigy0
                boffx = x0 - bigx0
                boffy = y0 - bigy0
                assert (bigw >= mw)
                assert (bigh >= mh)
                assert (boffx >= 0)
                assert (boffy >= 0)
                bigMask = np.zeros((bigh, bigw), bool)
                if modelMask.mask is not None:
                    bigMask[boffy:boffy + mh,
                            boffx:boffx + mw] = modelMask.mask
                else:
                    bigMask[boffy:boffy + mh, boffx:boffx + mw] = True
                bigMask = ModelMask(bigx0, bigy0, bigMask)
                # print('Recursing:', self, ':', (mh,mw), 'to', (bigh,bigw))
                bigmodel = self._realGetUnitFluxModelPatch(
                    img, px, py, minval, modelMask=bigMask)
                return Patch(x0, y0, bigmodel.patch[boffy:boffy + mh, boffx:boffx + mw])

        # get the FT of high-resolution PSF
        # (pxx5, pyx5) -- pixel coordinates in 5-times scaled-up image
        pxx_hires = (px + 0.5) * x_oversample - 0.5
        pyx_hires = (py + 0.5) * y_oversample - 0.5

        # P -- FT of PSF
        # (cx, cy) -- pixel coordinates of the PSF center
        # (pH, pW) -- height and width of PSF
        # (v, w) -- DFT frequencies long x- and y-axes
        P, (cx, cy), (pH, pW), (v, w) = psf.getFourierTransformHiRes(
            px, py, halfsize)

        # (dxx_hires, dyx_hires) -- pixel coordinates of the first pixel of PSF
        #                           in N-times scaled-up image
        dxx_hires = pxx_hires - cx
        dyx_hires = pyx_hires - cy
        if modelMask is not None:
            # The Patch we return *must* have this origin.
            ix0 = x0
            iy0 = y0
            # integer portions of (ix0, iy0) in oversample, scaled-up image
            ix0x_hires = int(np.round((ix0 + 0.5) * x_oversample - x_oversample/2.))
            iy0x_hires = int(np.round((iy0 + 0.5) * y_oversample - y_oversample/2.))
            # difference that we have to handle by shifting the 5-times
            # scaled-up model image
            muxx_hires = dxx_hires - ix0x_hires
            muyx_hires = dyx_hires - iy0x_hires
            # We will handle the integer portion by computing a shifted
            # image and copying it into the result.
            sxx_hires = int(np.round(muxx_hires))
            syx_hires = int(np.round(muyx_hires))
            # the subpixel portion will be handled with a Lanczos
            # interpolation
            muxx_hires -= sxx_hires
            muyx_hires -= syx_hires
        else:
            # integer portions of pixel coordinates
            dxx_hires = np.asarray(dxx_hires)
            dyx_hires = np.asarray(dyx_hires)

            if dxx_hires.size != 1 or dyx_hires.size != 1:
                raise ValueError(
                    f"Expected scalar high-res offsets, got "
                    f"dxx_hires shape={dxx_hires.shape}, dyx_hires shape={dyx_hires.shape}"
                )

            ix0x_hires = int(np.round(dxx_hires.ravel()[0]))
            iy0x_hires = int(np.round(dyx_hires.ravel()[0]))
            # subpixel portions of pixel coordinates
            muxx_hires = dxx_hires - ix0x_hires
            muyx_hires = dyx_hires - iy0x_hires
            # pixel coordinates of the first pixel of downscaled PSF in
            # real image
            ix0 = ix0x_hires // x_oversample
            iy0 = iy0x_hires // y_oversample
            #
            sxx_hires = syx_hires = 0

        # At this point, mux,muy are both in [-0.5, 0.5].
        assert abs(muxx_hires) <= 0.5
        assert abs(muyx_hires) <= 0.5

        wcs_ = NullWCS(pixscale=img.getWcs().pixscale / oversample)
        img_ = img.__class__(data=img.data,
                            inverr=img.inverr,
                            wcs=wcs_,
                            psf=img.psf,
                            sky=img.sky,
                            photocal=img.photocal,
                            name=img.name,
                            time=img.time)
        fftmix = self._getShearedProfile(img_, px, py)

        if fftmix is not None:
            Fsum = fftmix.getFourierTransform(v, w, zero_mean=True)
            G = np.fft.irfft2(Fsum * P)

            assert G.shape == (pH, pW)

            # Lanczos-3 interpolation in the same way we do for pixelized
            # PSFs.
            G = G.astype(np.float32)
            lanczos_shift_image(G, muxx_hires, muyx_hires, inplace=True)
        else:
            G = np.zeros((pH, pW), np.float32)

        # downscaling after padding zero-value pixels
        Gpad = np.zeros(oversample * np.ceil([s / oversample + 1 for s in G.shape]).astype(int),
                        dtype=G.dtype)

        pad_x0 = ix0x_hires % oversample + (sxx_hires % oversample)
        pad_y0 = iy0x_hires % oversample + (syx_hires % oversample)
        Gpad[pad_y0:pad_y0 + G.shape[0], pad_x0:pad_x0 + G.shape[1]] = G

        G = resize(Gpad, dsize=None, fx=1/oversample, fy=1/oversample, interpolation=INTER_AREA)
        G /= G.sum()

        if modelMask is not None:
            gh, gw = G.shape
            mh, mw = modelMask.shape

            sx = sxx_hires // oversample
            sy = syx_hires // oversample

            if gh % 2 == 0 and mh % 2 == 1:
                sx += 1
                ix0 -= 1
            if gw % 2 == 0 and mw % 2 == 1:
                sy += 1
                iy0 -= 1

            if sx != 0 or sy != 0:
                yi, yo = get_overlapping_region(-sy, -sy + mh - 1, 0, gh - 1)
                xi, xo = get_overlapping_region(-sx, -sx + mw - 1, 0, gw - 1)
                # shifted
                # FIXME -- are yo,xo always the whole image?  If so,
                # optimize
                shG = np.zeros((mh, mw), G.dtype)
                shG[yo, xo] = G[yi, xi]

                G = shG

            if gh > mh or gw > mw:
                G = G[:mh, :mw]

            assert G.shape == modelMask.shape
        else:
            # Clip down to suggested "halfsize"
            if x0 > ix0:
                G = G[:, x0 - ix0:]
                ix0 = x0
            if y0 > iy0:
                G = G[y0 - iy0:, :]
                iy0 = y0
            gh, gw = G.shape
            if gw + ix0 > x1:
                G = G[:, :x1 - ix0]
            if gh + iy0 > y1:
                G = G[:y1 - iy0, :]

        patch = Patch(ix0, iy0, G)

    else:
        patch = self._realGetUnitFluxModelPatch(img, px, py, minval, modelMask=modelMask, **kwargs)

    if patch is not None and modelMask is not None:
        assert patch.shape == modelMask.shape

    return patch
