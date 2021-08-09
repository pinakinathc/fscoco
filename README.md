------------------------
Dataset Paper:
------------------------

A retrieval framework for sketches possible multiple labels

List of experiments:

- Analysis on caption and sketch like ICCV reject work

- Homogenous CNN Top1 and Top10 for full sketch retrieval [Done]: top-1 19.57% top-10 47.82

- Heterogenous CNN Top1 and Top10 for full sketch retrieval[Done]: top-1 15.44% top-10 44.11%

- Top1 and Top10 for text retrieval

- Top1 and Top10 for joint sketch+text for image retrieval (multiplicative, additive, concatenation)

- Top1 and Top10 for 30%, 50%, 70% mask across temporal sequence front and back. This can empirically justify that important regions are drawn first.

- User adaptive sketch using meta-learning

- Perciever for sketch, text, or sketch+text


------------------------
Current Experiments:
------------------------

- version_0: Homogeneous CNN with Top1 and Top10 retrieval result

- version_1: Heterogenous CNN with Top1 and Top10 retrieval result

- version_2: Text-based Image Retrieval with GRU using our captions

- version_3: Text-based Image Retrieval with GRU using COCO captions

- version_4: Sketch+Text (concat) based Image Retrieval

- version_5: Sketch+Text (additive) based Image Retrieval

- version_6: Recap of version_0 to see if further improvement is possible

- version_7: checking version_0 using copy of code from version_4

- version_8: Text-based Image Retrieval with GRU with 2 num_layers


------------------------------------
Additional Experiments (wishlist):
------------------------------------

- (Benchmark) GCN for sketch and VGG for Image

- (Benchmark) ViT in homogenous and heterogenous fashion for sketches and images

- (Baseline Benchmark) 3 layer hierarchical LSTM for sketches and VGG for Images

- Check DALL-E for text to image generation and see if it can be adapted for sketch to image generation. [`Comment` -- seems to divert from the context of retrieval]


------------------
Analysis:
------------------

- Comparison of number of strokes and stroke length with Tu-Berlin, Sketchy, QuickDraw

- Stroke length with normalized time

- Uniqueness of n-grams of sketch captions and MS-COCO captions

- Percentage of Nouns, Verbs, Adjectives with sketch captions and MS-COCO captions

- Percentage of data and retrieval scores

- How complex are our reference images with respect to MS-COCO, i.e., how many things and amorphous background stuff?
