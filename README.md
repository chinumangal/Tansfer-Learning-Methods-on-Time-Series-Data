# Transfer Learning Methods on Time Series Data
---

## Description
Aim is to use transfer learning methods on time series data to predict values depending on the time series. In this special case we use CNC-data to predict the energy consumption of the code before it is executed.

---

## Dataset
Location: /dataset/Data/DMG_CMX_600V </br>
- The provided data is for two machine, DMG CMX 600V and DMC 60H. Basically it provides the movements of the different axis.
- At first we **only use the dataset from the DMG CMX 600V**
- The dataset is seperated in normal cuts and aircuts. The movements from both are the same, the difference is that there are no workpieces if it's and aircut and that the normal cuts have more parameter provided.
- The **parameterlist is available at /dataset/Data/ParameterDescription.txt** 
- The current for each axis is provided in the dataset and the models should be able to predict the current for movements, so that the needed power can be calculated
---

## Requirements
Model requirements: </br>
- predict the current usage of specific movements
- predict the current usage of whole CNC programs and depends further on:
    - normal cut or aircut
    - material
    - machine
---

## Project status 
Basic description on the progress of each model: </br>
- DTr-CNN:
    - derived from the DTr-CNN paper
    - CNN-architecture
    - can identify peaks in the current => identification also works for other materials
    - but no correct prediction possible with the amount of training
    - idea is to split up the high and low values into two models and determine in the prediction which model should be used for this datapoint
    - idea is to predcit a dataframe and only take the middle point into consideration => no summing the loss in the dataframe
- CNN:
---



