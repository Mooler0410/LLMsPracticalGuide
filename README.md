<h1 align="center">The Practical Guides for Large Language Models </h1>


<p align="center">
	<img src="https://camo.githubusercontent.com/64f8905651212a80869afbecbf0a9c52a5d1e70beab750dea40a994fa9a9f3c6/68747470733a2f2f617765736f6d652e72652f62616467652e737667" alt="Awesome" data-canonical-src="https://awesome.re/badge.svg" style="max-width: 100%;">	     
</p>

A curated (still actively updated) list of practical guide resources of LLMs. It's based on our survey paper: [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond](https://arxiv.org/abs/2304.13712).

These sources aim to help practitioners navigate the vast landscape of large language models (LLMs) and their applications in natural language processing (NLP) applications. If you find any resources in our repository helpful, please feel free to use them (and don't forget to cite our paper!)

## Latest News💥
- We used PowerPoint to plot the figure and released the source file [pptx](./source/figure_gif.pptx) for our GIF figure. We welcome pull requests to refine this figure, and if you find the source helpful, please cite our paper.

    ```bibtex
    @article{yang2023harnessing,
        title={Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond}, 
        author={Jingfeng Yang and Hongye Jin and Ruixiang Tang and Xiaotian Han and Qizhang Feng and Haoming Jiang and Bing Yin and Xia Hu},
        year={2023},
        eprint={2304.13712},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    ```

## Practical Guide for Models

We build an evolutionary tree of modern Large Language Models (LLMs) to trace the development of language models in recent years and highlights some of the most well-known models, in the following figure:

<p align="center">
<img width="600" src="./imgs/survey-gif-test.gif"/>
</p>



### BERT-style Language Models: Encoder-Decoder or Encoder-only

- BERT **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**, 2018, [Paper](https://aclanthology.org/N19-1423.pdf)
- RoBERTa **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**, 2019, [Paper](https://arxiv.org/abs/1909.11942)
- DistilBERT **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**, 2019, [Paper](https://arxiv.org/abs/1910.01108)
- ALBERT **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**, 2019, [Paper](https://arxiv.org/abs/1909.11942)
- ELECTRA **ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS**, 2020, [Paper](https://openreview.net/pdf?id=r1xMH1BtvB)
- T5 **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"**. *Colin Raffel et al.* JMLR 2019. [[Paper](https://arxiv.org/abs/1910.10683)]
- GLM **"GLM-130B: An Open Bilingual Pre-trained Model"**. 2022. [[Paper](https://arxiv.org/abs/2210.02414)] 
- AlexaTM **"AlexaTM 20B: Few-Shot Learning Using a Large-Scale Multilingual Seq2Seq Model"**. *Saleh Soltan et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2208.01448)]


### GPT-style Language Models: Decoder-only

- GPT-3 **"Language Models are Few-Shot Learners"**. NeurIPS 2020. [[Paper](https://arxiv.org/abs/2005.14165)]
- OPT **"OPT: Open Pre-trained Transformer Language Models"**. 2022. [[Paper](https://arxiv.org/abs/2205.01068)] 
- PaLM **"PaLM: Scaling Language Modeling with Pathways"**. *Aakanksha Chowdhery et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2204.02311)]
- BLOOM  **"BLOOM: A 176B-Parameter Open-Access Multilingual Language Model"**. 2022. [[Paper](https://arxiv.org/abs/2211.05100)]
- MT-NLG **"Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model"**. 2021. [[Paper](https://arxiv.org/abs/2201.11990)]
- GLaM **"GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"**. ICML 2022. [[Paper](https://arxiv.org/abs/2112.06905)]
- Gopher **"Scaling Language Models: Methods, Analysis & Insights from Training Gopher"**. 2021. [[Paper](http://arxiv.org/abs/2112.11446v2)]
- chinchilla **"Training Compute-Optimal Large Language Models"**. 2022. [[Paper](https://arxiv.org/abs/2203.15556)]
- LaMDA **"LaMDA: Language Models for Dialog Applications"**. 2021. [[Paper](https://arxiv.org/abs/2201.08239)]
- LLaMA **"LLaMA: Open and Efficient Foundation Language Models"**. 2023. [[Paper](https://arxiv.org/abs/2302.13971v1)]
- GPT-4 **"GPT-4 Technical Report"**. 2023. [[Paper](http://arxiv.org/abs/2303.08774v2)]
- BloombergGPT **BloombergGPT: A Large Language Model for Finance**, 2023, [Paper](https://arxiv.org/abs/2303.17564)
- GPT-NeoX-20B: **"GPT-NeoX-20B: An Open-Source Autoregressive Language Model"**. 2022. [[Paper](https://arxiv.org/abs/2204.06745)] 




## Practical Guide for Data


### Pretraining data
- **How does the pre-training objective affect what large language models learn about linguistic properties?**, ACL 2022. [Paper](https://aclanthology.org/2022.acl-short.16/)
- **Scaling laws for neural language models**, 2020. [Paper](https://arxiv.org/abs/2001.08361)
- **Data-centric artificial intelligence: A survey**, 2023. [Paper](https://arxiv.org/abs/2303.10158)
### Finetuning data
- **Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach**, EMNLP 2019. [Paper](https://arxiv.org/abs/1909.00161)
- **Language Models are Few-Shot Learners**, NIPS 2020. [Paper](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)
- **Does Synthetic Data Generation of LLMs Help Clinical Text Mining?** Arxiv 2023 [Paper](https://arxiv.org/abs/2303.04360)
### Test data/user data
- **Shortcut learning of large language models in natural language understanding: A survey**, Arxiv 2023. [Paper](https://arxiv.org/abs/2208.11857)
- **On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective** Arxiv, 2023. [Paper](https://arxiv.org/abs/2302.12095)





## Practical Guide for NLP Tasks
We build a decision flow for choosing LLMs or fine-tuned models~\protect\footnotemark for user's NLP applications. The decision flow helps users assess whether their downstream NLP applications at hand meet specific conditions and, based on that evaluation, determine whether LLMs or fine-tuned models are the most suitable choice for their applications.
<p align="center">
<img width="500" src="./imgs/decision.png"/>  
</p>

### Traditional NLU tasks

- **A benchmark for toxic comment classification on civil comments dataset** Arxiv 2023 [Paper](https://arxiv.org/abs/2301.11125)
- **Is chatgpt a general-purpose natural language processing task solver?** Arxiv 2023[Paper](https://arxiv.org/abs/2302.06476)
- **Benchmarking large language models for news summarization** Arxiv 2022 [Paper](https://arxiv.org/abs/2301.13848)
### Generation tasks
- **News summarization and evaluation in the era of gpt-3** Arxiv 2022 [Paper](https://arxiv.org/abs/2209.12356)
- **Is chatgpt a good translator? yes with gpt-4 as the engine** Arxiv 2023 [Paper](https://arxiv.org/abs/2301.08745)
- **Multilingual machine translation systems from Microsoft for WMT21 shared task**, WMT2021 [Paper](https://aclanthology.org/2021.wmt-1.54/)
- **Can ChatGPT understand too? a comparative study on chatgpt and fine-tuned bert**, Arxiv 2023, [Paper](https://arxiv.org/pdf/2302.10198.pdf)




### Knowledge-intensive tasks
- **Measuring massive multitask language understanding**, ICLR 2021 [Paper](https://arxiv.org/abs/2009.03300)
- **Beyond the imitation game: Quantifying and extrapolating the capabilities of language models**, Arxiv 2022 [Paper](https://arxiv.org/abs/2206.04615)
- **Inverse scaling prize**, 2022 [Link](https://github.com/inverse-scaling/prize)


### Abilities with Scaling

- **Training Compute-Optimal Large Language Models**, NeurIPS 2022 [Paper](https://openreview.net/pdf?id=iBBcRUlOAPR)
- **Scaling Laws for Neural Language Models**, Arxiv 2020 [Paper](https://arxiv.org/abs/2001.08361)
- **Solving math word problems with process- and outcome-based feedback**, Arxiv 2022 [Paper](https://arxiv.org/abs/2211.14275)
- **Chain of thought prompting elicits reasoning in large language models**, NeurIPS 2022 [Paper](https://arxiv.org/abs/2201.11903)
- **Emergent abilities of large language models**, TMLR 2022 [Paper](https://arxiv.org/abs/2206.07682)
- **Inverse scaling can become U-shaped**, Arxiv 2022 [Paper](https://arxiv.org/abs/2211.02011)


### Specific tasks
- **Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks**, Arixv 2022 [Paper](https://arxiv.org/abs/2208.10442)
- **PaLI: A Jointly-Scaled Multilingual Language-Image Model**, Arxiv 2022 [Paper](https://arxiv.org/abs/2209.06794)
- **AugGPT: Leveraging ChatGPT for Text Data Augmentation**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.13007)
- **Is gpt-3 a good data annotator?**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.10450)
- **Want To Reduce Labeling Cost? GPT-3 Can Help**, EMNLP findings 2021 [Paper](https://aclanthology.org/2021.findings-emnlp.354/)
- **GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation**, EMNLP findings 2021 [Paper](https://aclanthology.org/2021.findings-emnlp.192/)
- **LLM for Patient-Trial Matching: Privacy-Aware Data Augmentation Towards Better Performance and Generalizability**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.16756)
- **ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.15056)
- **G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.16634)
- **GPTScore: Evaluate as You Desire**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.04166)
- **Large Language Models Are State-of-the-Art Evaluators of Translation Quality**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.14520)
- **Is ChatGPT a Good NLG Evaluator? A Preliminary Study**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.04048)

### Real-World ''Tasks''


### Efficiency
1. Cost
- **Openai’s gpt-3 language model: A technical overview**, 2020. [Blog Post](https://lambdalabs.com/blog/demystifying-gpt-3)
- **Measuring the carbon intensity of ai in cloud instances**, ACM Conference on Fairness, Accountability, and Transparency 2022. [Paper](https://dl.acm.org/doi/abs/10.1145/3531146.3533234)
- **In AI, is bigger always better?**, Nature Article 2023. [Article](https://www.nature.com/articles/d41586-023-00641-w)
- **Language Models are Few-Shot Learners**, NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
- **Pricing**, OpenAI. [Blog Post](https://openai.com/pricing)
2. Latency
- **Holistic evaluation of language models**, Arxiv 2022. [Paper](https://arxiv.org/abs/2211.09110)
3. Parameter-Efficient Fine-Tuning
- **LoRA: Low-Rank Adaptation of Large Language Models**, Arxiv 2021. [Paper](https://arxiv.org/abs/2106.09685)
- **Prefix-Tuning: Optimizing Continuous Prompts for Generation**, ACL 2021. [Paper](https://aclanthology.org/2021.acl-long.353/)
- **P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks**, ACL 2022. [Paper](https://aclanthology.org/2022.acl-short.8/)
- **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**, Arxiv 2022. [Paper](https://arxiv.org/abs/2110.07602)


### Trustworthiness
1. Robustness and Calibration
- **Calibrate before use: Improving few-shot performance of language models**, ICML 2021. [Paper](http://proceedings.mlr.press/v139/zhao21c.html)
- **SPeC: A Soft Prompt-Based Calibration on Mitigating Performance Variability in Clinical Notes Summarization**, Arxiv 2023. [Paper](https://arxiv.org/abs/2303.13035)
  
2. Spurious biases
- **Shortcut learning of large language models in natural language understanding: A survey**, 2023 [Paper](https://arxiv.org/abs/2208.11857)
- **Mitigating gender bias in captioning system**, WWW 2020 [Paper](https://dl.acm.org/doi/abs/10.1145/3442381.3449950)
- **Calibrate Before Use: Improving Few-Shot Performance of Language Models**, ICML 2021 [Paper](https://arxiv.org/abs/2102.09690)
- **Shortcut Learning in Deep Neural Networks**, Nature Machine Intelligence 2020 [Paper](https://www.nature.com/articles/s42256-020-00257-z)
- **Do Prompt-Based Models Really Understand the Meaning of Their Prompts?**, NAACL 2022 [Paper](https://aclanthology.org/2022.naacl-main.167/)
  
3. Safety issues
- **GPT-4 System Card**, 2023 [Paper](https://cdn.openai.com/papers/gpt-4-system-card.pdf)
- **The science of detecting llm-generated texts**, Arxiv 2023 [Paper](https://arxiv.org/pdf/2303.07205.pdf)
- **Constitutional ai: Harmlessness from ai feedback**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.08073)
- **How stereotypes are shared through language: a review and introduction of the aocial categories and stereotypes communication (scsc) framework**, Review of Communication Research, 2019 [Paper](https://research.vu.nl/en/publications/how-stereotypes-are-shared-through-language-a-review-and-introduc)
- **Gender shades: Intersectional accuracy disparities in commercial gender classification**, FaccT 2018 [Paper](https://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf)

