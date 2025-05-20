# HyperTrans
This is the code for paper HyperTrans: Efficient Hypergraph-Driven Cross-Domain Pattern Transfer in Image Anomaly Detection.

## requirements
HyperTrans is implemented with Python 3.9.19, pip libs can be installed using:
```
pip install -r requirements.txt
```

## usage
You can use HyperTrans with 5-shot settings simply using 
```
python main.py --dataset_type [mvtec3d/eyecandies] --dataset_path [your_dataset_path] --shot 5
```

If you want to specify pre-extracted pattern banks, use
```
python main.py --bank_path .
```

Other settings can be seen in "main.py".

## authors
Tengyu Zhang, Deyu Zeng, Baoqiang Li, Wei Wang, Wei Liu, Zongze Wu

## about the paper
HyperTrans: Efficient Hypergraph-Driven Cross-Domain Pattern Transfer in Image Anomaly Detection,
IJCAI2025

[DOI](wait for publishing)

Abstract: Anomaly detection plays a pivotal role in industrial quality assurance processes, with cross-domain problems, exemplified by the model upgrade from RGB to 3D, being prevalent in real-world scenarios yet remaining systematically underexplored. To address the severe challenges posed by the extreme lack of datasets in target domain, we retain the knowledge from source models and explore a novel solution for anomaly detection through cross-domain learning, introducing HyperTrans. Targeting few-shot scenarios, HyperTrans centers around hypergraphs to model the relationship of the limited patch features and employs a perturbation-rectification-scoring architecture. The domain perturbation module injects and adapts channel-level statistical perturbations, mitigating style shifts during domain transfer. Subsequently, a residual hypergraph restoration module utilizes a cross-domain hypergraph to capture higher-order correlations in patches and align them across domains. Ultimately, with feature patterns exhibiting reduced domain shifts, an inter-domain scoring module aggregates similarity information between patches and normal patterns within the multi-domain subhypergraphs to make an integrated decision, generating multi-level anomaly predictions. Extensive experiments demonstrate that HyperTrans offers significant advantages in anomaly classification and anomaly segmentation tasks, outperforming state-of-the-art non-cross-domain methods in image-wise ROCAUC by 13%, 12%, and 15% in 1-shot, 2-shot, and 5-shot settings on MVTec3D AD.
Keywords: Transfer, Low-shot learning, Anomaly detection

## citations
If this code helps you, please cite:
```
wait for publishing
```

## other resources
We are happy to provide other resources such as figures, experimental results and code. Please email eziozhang7956@gmail.com.
