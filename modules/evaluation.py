import lpips

loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores


def calculate_lpips_score(source_image, target_image):
    d = loss_fn_alex(source_image, target_image)
    return d


def calculate_aed_score(source_image, target_image):
    return (source_image - target_image).pow(2).sum(3).sqrt()
