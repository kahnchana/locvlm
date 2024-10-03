# LocVLM
Unofficial Implementation of "Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs" presented at CVPR '24. 

[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Ranasinghe_Learning_to_Localize_Objects_Improves_Spatial_Reasoning_in_Visual-LLMs_CVPR_2024_paper.pdf)


## Abstract
>Integration of Large Language Models (LLMs) into visual domain tasks, resulting in visual-LLMs (V-LLMs), has enabled exceptional performance in vision-language tasks, particularly for visual question answering (VQA). However, existing V-LLMs (e.g. BLIP-2, LLaVA) demonstrate weak spatial reasoning and localization awareness. Despite generating highly descriptive and elaborate textual answers, these models fail at simple tasks like distinguishing a left vs right location. In this work, we explore how image-space coordinate based instruction fine-tuning objectives could inject spatial awareness into V-LLMs. We discover optimal coordinate representations, data-efficient instruction fine-tuning objectives, and pseudo-data generation strategies that lead to improved spatial awareness in V-LLMs. Additionally, our resulting model improves VQA across image and video domains, reduces undesired hallucination, and generates better contextual object descriptions. Experiments across 5 vision-language tasks involving 14 different datasets establish the clear performance improvements achieved by our proposed framework.


## Spatial Reasoning Evaluation
We add the COCO derived dataset, COCO-Spatial used in this paper for spatial evaluations. 
Refer to `dataset/coco_spatial_dataset.py` for loading this data. You will need to download images for _"COCO 2014 val images"_ set from [here](https://cocodataset.org/#download).
The filtered annotations for spatial QA pairs will be downloaded and generated automatically (~10 MB of downloads).
