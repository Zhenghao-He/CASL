import numpy as np
from PIL import Image

def add_gaussian_noise(
    img_input,
    noise_std=0.2,
    savefile=None,
    clip=True
):
    """
    给图像添加高斯噪声并可保存为PNG。

    参数：
    - img_input: str路径或numpy数组 (H,W,C)，值可为0~255或0~1
    - noise_std: float，高斯噪声标准差
    - savefile: str，若指定则保存为PNG文件
    - clip: 是否clip到[0,1]

    返回：
    - noisy_img: numpy数组，浮点型，范围0~1
    """
    # 加载图像
    if isinstance(img_input, str):
        img = np.array(Image.open(img_input).convert('RGB')).astype(np.float32) / 255.0
    else:
        img = img_input.astype(np.float32)
        if img.max() > 1.1:
            img /= 255.0

    # 添加噪声
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    noisy_img = img + noise
    if clip:
        noisy_img = np.clip(noisy_img, 0, 1)

    # 保存
    if savefile:
        Image.fromarray((noisy_img * 255).astype(np.uint8)).save(savefile)

    return noisy_img

# 示例：对图像加噪并保存


import numpy as np
from PIL import Image

def generate_noise_image(
    size=(256, 256),
    noise_std=1.0,
    savefile=None
):
    """
    生成一张纯高斯噪声图像。

    参数：
    - size: tuple (H, W)，图像尺寸
    - noise_std: float，高斯噪声标准差（越大越强）
    - savefile: str，保存路径（如'noise.png'），可选

    返回：
    - noise_img: numpy.ndarray，float32, 值范围[0,1]
    """
    # 生成高斯噪声（RGB）
    noise = np.random.normal(0.5, noise_std, (size[0], size[1], 3))
    noise = np.clip(noise, 0, 1)

    # 保存
    if savefile:
        Image.fromarray((noise * 255).astype(np.uint8)).save(savefile)

    return noise

generate_noise_image(
    size=(256, 256),
    noise_std=1.0,
    savefile='noise_output.png'
)
# noisy = add_gaussian_noise('/p/realai/zhenghao/LoRA_SAE/original.png', noise_std=0.25, savefile='noisy_output.png')






# ====== 用法举例 ======


