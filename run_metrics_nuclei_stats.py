# Calculate nuclei statistics metrics for evaluating virtual staining model: nuclei count, area, and aspect ratio
# Note: there should be multiple real and virtual H&E image pairs in the respective directories

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from skimage.color import rgb2hed
from histomicstk.features import compute_nuclei_features
from skimage.measure import label

def segment_nuclei_from_rgb(img_rgb):
    # 将 H&E 图转换为 hematoxylin 通道，简单阈值分割核
    hed = rgb2hed(img_rgb)
    h = hed[..., 0]
    # 使用 Otsu 等自定义阈值策略更优，此处以中位数为例
    thresh = np.median(h)
    mask = h < thresh
    label_mask = label(mask)
    return label_mask, h  # return segmented label mask and nuclei intensity

def get_nuclei_stats(im_label, im_nuclei):
    # compute morphometry features
    df = compute_nuclei_features(im_label, im_nuclei=im_nuclei,
                                 morphometry_features_flag=True,
                                 fsd_features_flag=False,
                                 intensity_features_flag=False,
                                 gradient_features_flag=False,
                                 haralick_features_flag=False)

    # select area and aspect ratio
    area = df['Size.Area']
    # aspect ratio: major_axis_length / minor_axis_length
    aspect = df['Size.MajorAxisLength'] / (df['Size.MinorAxisLength'] + 1e-6)
    aspect = np.clip(aspect, 0, 10)  # optionally clip to avoid extreme values
    return len(df), area.values, aspect.values

def nuclei_statistics_main(real_dir, fake_dir, output_dir):
    metrics = {'image': [], 'type': [], 'count': [], 'mean_area': [], 'mean_aspect': []}
    
    real_HE_list = [im_name for im_name in os.listdir(real_dir) if (im_name.endswith(".tif") or im_name.endswith(".tiff") or im_name.endswith(".png"))]
    fake_HE_list = [im_name for im_name in os.listdir(fake_dir) if (im_name.endswith(".tif") or im_name.endswith(".tiff") or im_name.endswith(".png"))]
    real_HE_list.sort()
    fake_HE_list.sort()
    assert len(real_HE_list) == len(fake_HE_list), "The number of real and virtual H&E images must be the same."
    print(f"Number of real H&E images: {len(real_HE_list)}, Number of virtual H&E images: {len(fake_HE_list)}")

    for i in range(len(real_HE_list)):
        real = imread(os.path.join(real_dir, real_HE_list[i]))[..., :3]
        fake = imread(os.path.join(fake_dir, fake_HE_list[i]))[..., :3]
        for typ, img in [('real', real), ('fake', fake)]:
            lbl, nuc = segment_nuclei_from_rgb(img)
            cnt, areas, aspects = get_nuclei_stats(lbl, nuc)
            metrics['image'].append(real_HE_list[i])
            metrics['type'].append(typ)
            metrics['count'].append(cnt)
            metrics['mean_area'].append(np.mean(areas) if len(areas)>0 else 0)
            metrics['mean_aspect'].append(np.mean(aspects) if len(aspects)>0 else 0)

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_dir, 'nuclei_statistics.csv'), index=False)
    print(df.groupby('type')[['count','mean_area','mean_aspect']].mean())

    # 绘图
    fig, axs = plt.subplots(1,3, figsize=(18,5))
    sns.boxplot(x='type', y='count', data=df, ax=axs[0])
    axs[0].set_title('Nuclei Count per Image')
    sns.boxplot(x='type', y='mean_area', data=df, ax=axs[1])
    axs[1].set_title('Mean Nucleus Area per Image')
    sns.boxplot(x='type', y='mean_aspect', data=df, ax=axs[2])
    axs[2].set_title('Mean Aspect Ratio per Image')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nuclei_statistics_plots.png'))
    plt.show()

if __name__ == '__main__':
    
    root_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides"
    input_TPAF_dir = os.path.join(root_dir, "01_og_TPAF_RGB_patches")
    real_HE_dir = os.path.join(root_dir, "01_og_real_HE_patches")
    virtual_HE_dir = os.path.join(root_dir, "05_results_vHE_patches_nuc_hi")

    nuclei_statistics_main(real_HE_dir, virtual_HE_dir, root_dir)
