---
layout: page
title: Resume
subtitle: Computer Science & Brain and Cognitive Sciences Student
author: Belay Zeleke
meta_description: "Resume - Belay Zeleke | KAIST CS & Brain Sciences | Machine Learning & AI Research"
keywords: "Belay Zeleke resume, KAIST, computer science, brain and cognitive sciences, machine learning, deep learning"
---

<div class="resume-container">

<div class="interest-card">
<h2>Interest</h2>
<h3>Machine Learning → Computer Vision → Neuroscience → Brain-Computer Interfaces</h3>
<p><em>Deep Learning, Neural Networks, AI Research, fMRI/EEG Signal Processing, Neural Decoding</em></p>
</div>

<div class="section-card">
<h2>Education</h2>

<div class="experience-item">
  <div class="company-header">
    <h3>Korea Advanced Institute of Science and Technology (KAIST)</h3>
  </div>
  <span class="role">Triple Major</span>
  <span class="date">August 2022 - Present</span>
  <p><strong>Major in Computer Science</strong> | <strong>Minor in Individually Designed AI Major</strong> | <strong>Double Major in Brain and Cognitive Sciences</strong></p>
  <p><em>Coursework:</em> Introduction to Deep Learning, Machine Learning for Computer Vision, Deep Learning for Computer Vision, AI and Brain, Data Structures, Discrete Mathematics, Systems Neuroscience, Biology of Neurons, Kaggle Computer Vision, Machine Learning for 3D Data, Introduction to Reinforcement Learning, Introduction to Cognitive Neuroscience, Database Systems</p>
</div>

</div>

<div class="section-card">
<h2>Labs and Internships</h2>

<div class="experience-item">
  <div class="company-header">
    <h3>Decision Brain Dynamics Lab, KAIST</h3>
  </div>
  <span class="role">Research Intern</span>
  <span class="date">July 2025 - Present</span>
  <a href="https://raphe.kaist.ac.kr" class="project-link">Lab Website</a>
  <ul>
    <li>Conducting fMRI to Natural Language Decoding research, focusing on non-invasive brain-to-language translation models</li>
    <li>Working on improving semantic and temporal alignment between EEG signals and linguistic embeddings for continuous language reconstruction</li>
    <li>Developing deep learning pipelines integrating transformer-based semantic decoders, temporal convolutional encoders, and cross-modal attention mechanisms for robust neural decoding</li>
    <li>Exploring multi-region EEG signal representations (frontal, parietal, occipital) to model distributed cortical contributions to semantic comprehension and imagined speech decoding</li>
    <li>Aiming to advance non-invasive brain-computer interfaces capable of reconstructing internal thoughts and perceived speech in real time</li>
  </ul>
</div>

<div class="experience-item">
  <div class="company-header">
    <h3>Multi-Modal AI Lab, KAIST</h3>
  </div>
  <span class="role">Research Intern</span>
  <span class="date">Dec 2024 - Aug 2025</span>
  <a href="https://mm.kaist.ac.kr" class="project-link">Lab Website</a>
  <p>Implemented and presented many S.O.T.A generative models, including:</p>
  <ul>
    <li>VAE, β-VAE, GAN, VQ-VAE, VQ-GAN</li>
    <li>StyleGAN-1 & 2, WGAN, CycleGAN, StarGAN, WGAN-GP</li>
    <li>DDPM (Denoising Diffusion Probabilistic Model), DDIM</li>
    <li>Diffusion Models Beat GANs on Image Synthesis</li>
    <li>Classifier-Free Diffusion Guidance</li>
  </ul>
</div>

</div>

<div class="section-card">
<h2>Projects</h2>

<div class="project-item">
  <h4>Neural Machine Translation with LSTM, Attention, and Transformer Models</h4>
  <div class="tech-stack">PyTorch, NLP, LSTM, Transformers, Attention</div>
  <ul>
    <li>Implemented Neural Machine Translation models using LSTM, Attention-based LSTM, and Transformer architectures on the Multi30k dataset (German-English)</li>
    <li>Custom implementations of LSTMCell, Multi-head Attention, achieving competitive BLEU scores</li>
  </ul>
  <a href="https://github.com/belay-cell/Neural-Machine-Translation" class="project-link">GitHub</a>
</div>

<div class="project-item">
  <h4>Face Recognition Using PCA, LDA, Ensemble Learning and Randomized Forests</h4>
  <div class="tech-stack">Python, Machine Learning, PCA, LDA, Random Forest</div>
  <ul>
    <li>Developed face recognition models using PCA, LDA, and ensemble learning techniques</li>
    <li>Explored dimensionality reduction, generative-discriminative subspace learning, and advanced classification methods</li>
    <li>Achieved high recognition accuracy through optimized PCA-LDA ensembles and Random Forest parameters</li>
  </ul>
  <a href="https://github.com/belay-cell/Face-Recognition" class="project-link">GitHub</a>
</div>

<div class="project-item">
  <h4>Semantic Segmentation using Fully Convolutional Network (FCN)</h4>
  <div class="tech-stack">PyTorch, Computer Vision, FCN, PASCAL VOC</div>
  <ul>
    <li>Performed semantic segmentation on PASCAL VOC 2011 dataset with 20 object categories</li>
    <li>Utilized Semantic Boundaries Dataset (SBD) for enhanced segmentation performance</li>
    <li>Classified CIFAR10 images into 10 categories using Convolutional Neural Networks</li>
  </ul>
  <a href="https://github.com/belay-cell/Semantic-Segmentation-FCN" class="project-link">GitHub</a>
</div>

<div class="project-item">
  <h4>VQ-GAN (Vector-Quantized GAN)</h4>
  <div class="tech-stack">PyTorch, GANs, VQGAN, Transformers</div>
  <p>Implemented two-stage VQGAN Transformer model that learns discrete image codebook with adversarial training and autoregressively models latent tokens for high-resolution image synthesis, following Esser et al. (CVPR 2021)</p>
  <a href="https://github.com/belay-cell/VQ-GAN" class="project-link">GitHub</a>
</div>

<div class="project-item">
  <h4>Deep Face Recognition with Metric Learning</h4>
  <div class="tech-stack">PyTorch, ResNet18, Triplet Loss, Mixed Precision</div>
  <ul>
    <li>Designed lightweight face recognition system (11.5M parameters) using ResNet18</li>
    <li>Developed flexible training framework supporting Softmax and Triplet loss functions</li>
    <li>Implemented mixed-precision training and multi-GPU support for optimized performance</li>
  </ul>
  <a href="https://github.com/belay-cell/Deep-Face-Recognition-Metric-Learning" class="project-link">GitHub</a>
</div>

<div class="project-item">
  <h4>ML for 3D Data - PointNet: Deep Learning on Point Cloud Data</h4>
  <div class="tech-stack">PyTorch, 3D Vision, PointNet, ModelNet40</div>
  <p>Implemented PointNet-based architectures for supervised 3D point cloud classification, part segmentation, and auto-encoding on ModelNet40 and ShapeNet Part datasets, following Qi et al. framework</p>
  <a href="https://github.com/belay-cell/Point-Cloud-Classification-and-Segmentation" class="project-link">GitHub</a>
</div>

<div class="project-item">
  <h4>NeRF: Neural Radiance Fields for 3D Scene Reconstruction</h4>
  <div class="tech-stack">PyTorch, 3D Vision, NeRF, Volumetric Rendering</div>
  <p>Implemented Neural Radiance Fields (NeRF) for novel 3D view synthesis from 2D images, following Ben Mildenhall et al. (ECCV 2020). Trained and rendered complex scenes using volumetric rendering</p>
  <a href="https://github.com/belay-cell/NeRF-3D-Scene-Reconstruction" class="project-link">GitHub</a>
</div>

<div class="project-item">
  <h4>Vector Database (ChromaDB, LangChain RAG)</h4>
  <div class="tech-stack">Python, LangChain, ChromaDB, HuggingFace, RAG</div>
  <ul>
    <li>Implemented RAG systems using ChromaDB and LangChain for intelligent document question-answering</li>
    <li>Featured HuggingFace embeddings (all-MiniLM-L6-v2) for semantic search and LLM-based contextual answer generation</li>
    <li>Built production-ready applications: technical documentation QA system and OpenBookQA scientific knowledge retrieval platform</li>
    <li>Applied advanced NLP techniques including recursive text chunking, dense vector embeddings, and similarity search algorithms</li>
  </ul>
  <a href="https://github.com/belay-cell/Vector-Database-ChromaDB-LangChain-RAG" class="project-link">GitHub</a>
</div>

</div>

<div class="section-card">
<h2>Awards and Certificates</h2>

<div class="experience-item">
  <h4>Gold Honor in International Youth Math Challenge-Olympiad (IYMC)</h4>
  <p><strong>National Award of Ethiopia</strong></p>
</div>

</div>

<div class="section-card">
<h2>Skills</h2>

<div class="experience-item">
  <p><strong>Programming Languages:</strong> PyTorch, TensorFlow, Python, Java</p>
  <p><strong>Tools and Technologies:</strong> MATLAB, Deep Learning, Machine Learning, HTML/CSS, Git, Docker</p>
</div>

</div>

<style>
.resume-container { max-width: 900px; margin: 0 auto; }
.interest-card, .section-card { 
  background: #f8f9fa; 
  border-left: 4px solid #0085A1; 
  padding: 20px; 
  margin: 20px 0; 
  border-radius: 5px; 
}
.interest-card h2, .section-card h2 { 
  color: #0085A1; 
  margin-top: 0; 
}
.interest-card h3 { 
  color: #333; 
  font-size: 1.3em; 
  margin: 10px 0; 
}
.experience-item, .project-item { 
  margin: 25px 0; 
  padding-bottom: 15px; 
  border-bottom: 1px solid #dee2e6; 
}
.experience-item:last-child, .project-item:last-child {
  border-bottom: none;
}
.company-header h3 { 
  color: #333; 
  margin: 0; 
  font-size: 1.2em; 
}
.role { 
  font-weight: 600; 
  color: #0085A1; 
}
.date { 
  color: #6c757d; 
  font-style: italic; 
  margin-left: 10px;
}
.project-item h4 { 
  color: #333; 
  margin: 0 0 10px 0; 
}
.tech-stack { 
  background: #e9ecef; 
  padding: 5px 10px; 
  border-radius: 3px; 
  font-size: 0.9em; 
  color: #495057; 
  display: inline-block; 
  margin-bottom: 10px; 
}
.project-link { 
  color: #0085A1; 
  text-decoration: none; 
  font-weight: 600; 
}
.project-link:hover { 
  text-decoration: underline; 
}
</style>

</div>
