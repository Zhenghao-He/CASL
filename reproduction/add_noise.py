import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    给图片添加高斯噪声

    :param image: 输入的彩色/灰度图片（numpy数组）
    :param mean: 高斯噪声均值
    :param sigma: 高斯噪声标准差
    :return: 加噪后的图片
    """
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise
    # 保证像素范围在[0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# 加载图片（你可以改成自己的图片路径）
img = cv2.imread('/p/realai/zhenghao/LoRA_SAE/ffhq-dataset/images1024x1024/00000/00678.png')
# img = cv2.imread('/p/realai/zhenghao/LoRA_SAE/ffhq-dataset/images1024x1024/00000/00568.png')
# 如果是彩色，BGR转RGB（可选）
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 加高斯噪声
noisy_img = add_gaussian_noise(img, mean=0, sigma=0)

# 显示图片（需要matplotlib）
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.title('Original')
plt.imshow(img)
plt.axis('off')

plt.subplot(1,2,2)
plt.title('With Gaussian Noise')
plt.imshow(noisy_img)
plt.axis('off')
plt.show()

# 保存结果
img_small = cv2.resize(noisy_img, (256, 256))
cv2.imwrite('noisy_output.jpg', cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR))
