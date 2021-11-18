import numpy as np

from .ssim import ssim


def ignore_boundary(img, SCALE):
    h, w = img.shape[:2]
    boundarypixels = 6 + SCALE 
    img = img[:h-h%SCALE, :w-w%SCALE]
    img = img[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels]
    return img

def best_psnr(img_orig, img_out):

    SCALE = 4
    SHIFT = 40
    SIZE = 30

    img_orig = ignore_boundary(img_orig, SCALE)
    img_out = ignore_boundary(img_out, SCALE)

    h, w = img_orig.shape[:2]
    c = img_orig.shape[2] if len(img_orig.shape) == 3 else 1
    h_cen, w_cen = int(h / 2), int(w / 2)
    h_left = h_cen - SIZE
    h_right = h_cen + SIZE
    w_left = w_cen - SIZE
    w_right = w_cen + SIZE

    im_h = img_orig[None, h_left:h_right, w_left:w_right]
    ssim_h = img_orig[h_left:h_right, w_left:w_right]


    im_shifts = np.zeros([(2 * SHIFT + 1) * (2 * SHIFT + 1), *ssim_h.shape])
    ssim_shifts = np.zeros([(2 * SHIFT + 1) * (2 * SHIFT + 1), c])
    for hei in range(-SHIFT, SHIFT + 1):
        for wid in range(-SHIFT, SHIFT + 1):
            tmp_l = img_out[h_left + hei:h_right + hei, w_left + wid:w_right + wid]
            im_shifts[(hei + SHIFT) * (SHIFT + 1) + wid + SHIFT, :, :] = tmp_l

	        #ssim_h = np.squeeze(im_h)
            # ssim_h = ssim_h.astype('uint8')
            # ssim_l = tmp_l.astype('uint8')
            if abs(hei) % 2 == 0 and abs(wid) % 2 == 0:
                if c == 1:
                    ssim_shifts[(hei + SHIFT) * (SHIFT + 1) + wid + SHIFT, 0] \
                        = ssim(tmp_l[:, :], ssim_h[:, :])
                else:
                    for i in range(c):
                        ssim_shifts[(hei + SHIFT) * (SHIFT + 1) + wid + SHIFT, i] \
                            = ssim(tmp_l[:, :, i], ssim_h[:, :, i])

    squared_error = np.square(im_shifts / 255. - im_h / 255.)
    mean_aixs = (1, 2, 3) if c == 3 else (1, 2)
    mse = np.mean(squared_error, axis=mean_aixs)
    psnr = 10 * np.log10(1.0 / mse)
    return max(psnr), max(np.mean(ssim_shifts, axis=1))