Logan Whitehouse
CONOS Paper Writeup


1. Please explain in 2 sentences or less what the problem being solved is.
    - Under a lot of experimental conditions cells are being sampled from many differenc sources, which leads to the issue of being able to effectively combine all the data about cell types and relationships across samples that may not be at all consistent between samples.

2. What were the main contributions of the authors in this work? (You can answer in a few bullet points).
    - Develop a "unified graph representation" that is able to relate cells across samples, effectively making one large, cross-sample graph.
    - Explore the ability of this type of joint graph structure to analyze heterogeneous samples across a series of experiments.
    - Give a very good case for adapting this type of approach for doing cross-sample studies, even including the ability to provide output for further RNA-seq analysis software.

3. Please describe 1-2 computational experiments that the authors implemented to test their method.
    - Reanalyzed the Taula Muris mouse atlas to combine 48 sets of tissue data, combined this with another atlas to create a huge joint graphh that was able to identify cell populations across samples (and sequencing platforms).
    - Assembled a mouse single-cell chromatin accessibility atlas using multi-sample chromatin accessibility graphs.

4. Were the authors the first to attempt this particular problem? If not, did they compare their results to other baselines? Do you think that their evaluation was objective?
   - As far as I can tell they were the first to do multi-sample analysis in this precise way, by joint graph structure. They do, however, compare it to other tools that attempt to do similar things (Seurat and Scran, fig.2) and by all accounts it looks like CONOS beats them in all metrics shown here. I don't know enough about the best metrics to measure these by (they could very easily have just shown the good ones) but I'd believe it to be objective enough.

5. Do you think that the authors provided enough evidence for why their developed method is an important contribution? If yes, please describe their reasoning here. If you do not think they adequately justified why they worked on this particular problem, please describe your thoughts on that here.
   - I think they both implicitly and explicitly gave plenty of evidence. Off the bat it's easy to see why it's important in the field of cross-sample analysis, especially with the wide variety of sequencing platforms, experimental conditions, and historical data out there. Allowing for cross-sample studies clearly allows for more robust research and new results that otherwise could've been missed by lesser methods. Explicitly they show some very good use cases (the mouse cell atlas and large cross sample study) and I think that justifies itself.

6. What is one follow-up idea or extension from this work?
    - My first follow up idea is to test the limits of how many/how variable we could go with the samples and still get relatively clean output. Say, take some drastically different experiments that produced sc data and see what the extremes of the algorithm are. Probably would be hard to benchmark, but would be cool to know the limits. 