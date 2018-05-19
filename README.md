# Multimodal sentiment analysis with multi-task learning

**Unimodal and multimodal uni-task, bi-task, and tri-task learning models for sentiment analysis using the CMU-MOSI database.**

* In uni-task models we perform regression experiments to predict sentiment scores.

* In bi-task models we perform multi-task learning experiments which have sentiment score regression as the main task, and intensity **or** polarity classification as the auxiliary task.

* In tri-task models we perform multi-task learing experiments which have sentiment score regression as the main task, and intensity **and** polarity classification as the auxiliary tasks.

* In multimodal models we compare Early Fusion, Late Fusion, Hierarchical Fusion, and Tensor Fusion Network.

**These codes are for our ACL2018 Computational Modeling of Human Multimodal Language Workshop paper:**

```latex
@inproceedings{tian2018polarity,
  title={Polarity and Intensity: the Two Aspects of Sentiment Analysis},
  author={Tian, Leimin and Lai, Catherine and Moore, Johanna},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  pages={to appear},
  year={2018}
}
```

"acl2018_appendix.pdf" contains results on the training and validation sets of the CMU-MOSI database, as an appendix to the test results reported in the paper.

The Hierarchical Fusion model in the multimodal experiments is based on our previous work:

```latex
@inproceedings{tian2016recognizing,
  title={Recognizing emotions in spoken dialogue with hierarchically fused acoustic and lexical features},  
  author={Tian, Leimin and Moore, Johanna and Lai, Catherine}, 
  booktitle={Spoken Language Technology Workshop (SLT), 2016 IEEE},
  pages={565--572},  
  year={2016},  
  organization={IEEE}
}
```
