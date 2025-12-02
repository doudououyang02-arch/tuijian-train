import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from FlagEmbedding import BGEM3FlagModel
from collections import Counter
import copy

# 加载BGE-M3模型
model = BGEM3FlagModel("/mnt/vdb2t_1/sujunyan/program/ui/image_features_extract/models/baai_bge-m3/baai_bg3-m3", use_fp16=False)

# 路径设置
image_folder = "/mnt/vdb2t_1/sujunyan/label30000/Pattern_recognition_filter/7000_results/final_good"
json_folder = "/mnt/vdb2t_1/sujunyan/label30000/Pattern_recognition_filter/7000_results/final_good_json"

# 获取图像文件路径和对应的json文件路径
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
json_files = [f.replace('.jpg', '.json').replace('.png', '.json').replace('.jpeg', '.json') for f in image_files]

# 用于存放所有图像描述和图像路径
image_paths = []
image_descriptions = []

# 权重设定（可以根据需求进行调整）
weights = {
    "design_semantics": 0.3,
    "color_semantics": 0.2,
    "layout_semantics": 0.2,
    "content_semantics": 0.3,
    "visual_features": 0.3,
    "font": 0.2,
    "layout": 0.2,
    "color": 0.2,
    "design_style": 0.3,
    "content_understanding": 0.4,
    "content": 0.3,
    "main_component": 0.2,
    "customized_component": 0.1,
    "image_component": 0.1,
    "icon_component": 0.1
}

# 可选：标准化权重使其加和等于 1
normalize_weights = False
if normalize_weights:
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}  # 将权重标准化

count_ = 0
# 2. 遍历所有图像文件和对应的JSON文件，提取文本描述并计算文本向量
for json_file, image_file in zip(json_files, image_files):
    
    count_ += 1
    print(f"第{count_}/{len(json_files)}个json文件")
    # if count_ > 10:
    #     break
    json_path = os.path.join(json_folder, json_file)
    image_path = os.path.join(image_folder, image_file)

    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取各部分文本描述
    design_semantics = data["semantic_pattern"]["Design_semantics"][0]["design_semantics"]
    color_semantics = data["semantic_pattern"]["Design_semantics"][0]["color_semantics"]
    layout_semantics = data["semantic_pattern"]["Design_semantics"][0]["layout_semantics"]
    content_semantics = data["semantic_pattern"]["Design_semantics"][0]["content_semantics"]
    visual_features = data["semantic_pattern"]["Visual_features"][0]["visual_features"]
    font = data["semantic_pattern"]["Visual_features"][0]["font"]
    layout = data["semantic_pattern"]["Visual_features"][0]["layout"]
    color = data["semantic_pattern"]["Visual_features"][0]["color"]
    design_style = data["semantic_pattern"]["Visual_features"][0]["design_style"]
    content_understanding = data["semantic_pattern"]["Content_understanding"][0]["content_understanding"]
    content = data["semantic_pattern"]["Content_understanding"][0]["content"]
    main_component = data["semantic_pattern"]["Content_understanding"][0]["component"][0]["main_component"]
    customized_component = data["semantic_pattern"]["Content_understanding"][0]["component"][0]["customized_component"]
    image_component = data["semantic_pattern"]["Content_understanding"][0]["component"][0]["image_component"]
    icon_component = data["semantic_pattern"]["Content_understanding"][0]["component"][0]["icon_component"]


    weights_copy = copy.deepcopy(weights)
    if image_component == "" or image_component == "无" or image_component == None:
        weights_copy["image_component"] = 0.0
    if icon_component == "" or icon_component == "无" or icon_component == None:
        weights_copy["icon_component"] = 0.0
    
    
    # 将描述向量乘以对应的权重
    weighted_description = (
        weights_copy["design_semantics"] * model.encode([design_semantics], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["color_semantics"] * model.encode([color_semantics], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["layout_semantics"] * model.encode([layout_semantics], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["content_semantics"] * model.encode([content_semantics], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["visual_features"] * model.encode([visual_features], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["font"] * model.encode([font], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["layout"] * model.encode([layout], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["color"] * model.encode([color], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["design_style"] * model.encode([design_style], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["content_understanding"] * model.encode([content_understanding], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["content"] * model.encode([content], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["main_component"] * model.encode([main_component], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["customized_component"] * model.encode([customized_component], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["image_component"] * model.encode([image_component], return_dense=True, max_length=8192)["dense_vecs"] +
        weights_copy["icon_component"] * model.encode([icon_component], return_dense=True, max_length=8192)["dense_vecs"]
    )

    # 保存图像路径和描述向量
    image_paths.append(image_path)
    image_descriptions.append(weighted_description)

# 将所有编码合并成一个矩阵
embeddings_matrix = np.concatenate(image_descriptions, axis=0)

# 保存到本地
np.save("text_embeddings_8820.npy", embeddings_matrix)
    
print(f"Extracted embeddings from {len(image_paths)} images and saved to 'text_embeddings_8820.npy'.")

path_matrix = np.stack(image_paths, axis=0)

np.save("path_matrix_8820.npy", path_matrix)

print(embeddings_matrix.shape)
print(path_matrix.shape)
