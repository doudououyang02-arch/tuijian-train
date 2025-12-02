# 微调文本嵌入模型
我需要微调一个文本嵌入模型，目前我有这个模型对于文本特征提取的代码（text_embedding.py）和根据文本相似度进行推荐的代码的代码（infer.py），你好好读一下这两个功能代码。

但是，老板那里说文本嵌入模型不能使用现成的，得进行一下微调。因此，我需要对这个模型进行微调训练。数据集在服务器的这个文件夹里面/mnt/vdb2t_1/sujunyan/label30000/Pattern_recognition_filter/7000_results/final_good_json，里面存放的都是json文件，json文件的内容说明如下所示：
```
{
        "design_semantics": 对应的文本内容,
        "color_semantics": 对应的文本内容,
        "layout_semantics": 对应的文本内容,
        "content_semantics": 对应的文本内容,
        "visual_features": 对应的文本内容,
        "font": 对应的文本内容,
        "layout": 对应的文本内容,
        "color": 对应的文本内容,
        "design_style": 对应的文本内容,
        "content_understanding": 对应的文本内容,
        "content": 对应的文本内容,
        "main_component": 对应的文本内容,
        "customized_component": 对应的文本内容,
        "image_component": 对应的文本内容,
        "icon_component": 对应的文本内容
    }
```

训练时采用的模型采用双塔式模型，分为a和b两部分，他们两个都由文本嵌入模型+全连接层组成。他们两个的文本嵌入模型参数是共享的，全连接层不共享。在训练时，会随机取json的某几个键，取的键的个数是可以提前定义的，方便调试。然后这几个键对应的文本值送入模型a，并对其进行随机掩码mask，mask比例10-95不等，mask后的文本送入模型b，想要达到的效果是a模型的输出与b模型的输出计算文本相似度，使得他们两个越接近越好。训练完成后帮我把微调后的最好的模型保存到本地，并更改infer.py这个代码让我进行推断，查看基于文本推荐的图像好不好用。
