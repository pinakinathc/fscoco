<a href="http://sketchx.ai/"><img src="https://pinakinathc.github.io/assets/images/organizations/sketchx-logo.png" style="width:70%; max-width: 300px; border:None;" class='img-fluid img-thumbnail'></a><br/><br/>

## **FS-COCO: Towards Understanding of Freehand Sketches of Common Objects in Context.**

---

### **Authors**
**[Pinaki Nath Chowdhury](https://pinakinathc.me), [Aneeshan Sain](https://aneeshan95.github.io/), [Ayan Kumar Bhunia](https://ayankumarbhunia.github.io/), [Tao Xiang](https://scholar.google.com/citations?user=MeS5d4gAAAAJ&hl=en), [Yulia Gryaditskaya](https://yulia.gryaditskaya.com/), [Yi-Zhe Song](https://scholar.google.co.uk/citations?user=irZFP_AAAAAJ&hl=en)**

SketchX, Center for Vision Speech and Signal Processing

University of Surrey, United Kingdom

**Published at ECCV 2022**

[Paper](http://www.pinakinathc.me/assets/papers/fscoco.pdf) / [Github](https://github.com/pinakinathc/fscoco)

### **Abstract**
We advance sketch research to scenes with the first dataset of freehand scene sketches, FS-COCO. With practical applications in mind, we collect sketches that convey well scene content but can be sketched within a few minutes by a person with any sketching skills. Our dataset comprises 10,000 freehand scene vector sketches with per point space-time information by 100 non-expert individuals, offering both object- and scene-level abstraction. Each sketch is augmented with its text description. Using our dataset, we study for the first time the problem of fine-grained image retrieval from freehand scene sketches and sketch captions. We draw insights on: (i) Scene salience encoded in sketches using the strokes temporal order; (ii) Performance comparison of image retrieval from a scene sketch and an image caption; (iii) Complementarity of information in sketches and image captions, as well as the potential benefit of combining the two modalities. In addition, we extend a popular vector sketch LSTM-based encoder to handle sketches with larger complexity than was supported by previous work. Namely, we propose a hierarchical sketch decoder, which we leverage at a sketch-specific “pre-text” task. Our dataset enables for the first time research on freehand scene sketch understanding and its practical applications

### **Dataset Statistics**

For our dataset, we compute two estimates of the category distribution across our data: (1) Upper Bound: based on semantic segmentation labels in images and (2) Lower Bound: based on the occurrence of a word in a sketch caption. 

<table>
    <tr>
        <th>Total Sketches</th>
        <th># Categories</th>
        <th colspan="4"># Categories per Sketch</th>
        <th colspan="4"># Sketches per Category</th>
    </tr>
    <tr>
        <td colspan="2"></td>
        <td>Mean</td>
        <td>Std</td>
        <td>Min</td>
        <td>Max</td>
        <td>Mean</td>
        <td>Std</td>
        <td>Min</td>
        <td>Max</td>
    </tr>
    <tr>
        <td>10,000</td>
        <td>92/150</td>
        <td>1.37/7.17</td>
        <td>0.57/3.27</td>
        <td>1/1</td>
        <td>5/25</td>
        <td>99.42/413.18</td>
        <td>172.88/973.59</td>
        <td>1/1</td>
        <td>866/6789</td>
    </tr>
</table>

### **Dataset Sample and Comparison with existing dataset.**
![Sample Comparison FSCOCO dataset](/assets/fscoco-sample-comparison.jpg)

## **Code**

### >> Code for Data collection tool
[**https://github.com/pinakinathc/SketchX-SST**](https://github.com/pinakinathc/SketchX-SST)

You will need to install `npm`, `nodejs`, `mongodb`, `pymongo`, `numpy`

Once you have done the setup, you are ready to run the code. I used a Linode server to host this service. It takes £5 for a month.

```
npm install
python init_db.py
sudo node server.js
```

Once you are done collecting data, you can visualise your results at scale using `python visualise_sketch.py`. You will need to install `bresenham` and `cv2` for this. Also modify Line `60, 61` to set the path of **MSCOCO** data directory and **SketchyCOCO** data directory.

### >> Code for running experiments

[**https://github.com/pinakinathc/fscoco**](https://github.com/pinakinathc/scene-sketch-dataset)

I use `PyTorch` and `PyTorch Lightning` for the experiments. If you face some issues with dependencies, please contant me.

I also added some code to run the experiments using HPC (i.e., Condor).

Example for running:
```
git clone https://github.com/pinakinathc/scene-sketch-dataset.git
cd scene-sketch-dataset/src/sbir_baseline
python main.py
```

Before you run `main.py` ensure that the code is set up in training mode and data path are correct in `options.py`.

### **License / Terms of Use**
Downloading this dataset means you agree to the following License / Terms of Use:

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

### **How to cite this dataset**
```
@inproceedings{fscoco,
    title={FS-COCO: Towards Understanding of Freehand Sketches of Common Objects in Context.}
    author={Chowdhury, Pinaki Nath and Sain, Aneeshan and Bhunia, Ayan Kumar and Xiang, Tao and Gryaditskaya, Yulia and Song, Yi-Zhe},
    booktitle={ECCV},
    year={2022}
}
```

### **Download this dataset**

[**[Official Storage]**](http://cvssp.org/data/fscoco/fscoco.tar.gz)
[**[Backup Storage]**](https://drive.google.com/file/d/1sjWoONedi9PBK4aQFNJUEL7diFJ7FyIN/view?usp=sharing)

### **Acknowledgements**
This dataset would not be possible without the support of the following wonderful people:

**[Anran Qi](https://anranqi.github.io/), [Yue Zhong](http://sketchx.ai/people), [Lan Yang](http://sketchx.ai/people), [Dongliang Chang](https://scholar.google.com/citations?user=tIf50PgAAAAJ&hl=en), [Ling Luo](https://rowl1ng.com/), [Ayan Das](https://scholar.google.com/citations?user=x-WI_EgAAAAJ&hl=en), [Zhiyu Qu](http://sketchx.ai/people), [Yixiao Zheng](http://sketchx.ai/people), [Ruolin Yang](http://sketchx.ai/people), [Ranit](https://github.com/MaestroRon-001)**
