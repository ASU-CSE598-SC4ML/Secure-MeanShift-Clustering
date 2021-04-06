# crypten_ms
Crypten MeanShift Algorithm


## Requirement
crypten, scikit-learn, matplotlib

### Create Environment from yml. 

    conda env create -f env/environment.yml

    conda activate test_crypten

***

### Module 1

1. Generate random centers in a [0, 1], [0, 1] plane, for 4 centers.

2. Generate random clustered point, using gaussian distribution around each centers, 250 points for each center (marked by different color).

3. Use scikit-learn to get the ground-truth centroid (marked by X).

![Example of Module 1](images/fig_generate.PNG)

### Module 2

1. Randomly sample a small set of points

2. Calculate the distance between point and dusts