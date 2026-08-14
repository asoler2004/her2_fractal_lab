# Color Deconvolution
Ruifrok–Johnston Color Deconvolution is a classic image processing technique introduced by Arnout Ruifrok and Dennis Johnston (2001) to separate (unmix) multiple histological stains in digital brightfield microscopy images.  It solves a key problem in digital pathology: when tissues are treated with multiple dyes—such as Hematoxylin (blue/purple) and DAB or Eosin (brown/pink)—their light absorption overlaps in standard RGB space. Color deconvolution isolates the optical density contribution of each individual stain so that scientists can quantify expression levels independently. 
Key Principles & Mechanics

1. The Beer–Lambert Law Basis:
Standard RGB image acquisition records transmitted light intensity (I) relative to the light source intensity (I_0). Because dyes absorb light subtractively rather than additively, intensity values are non-linear with respect to stain concentration.  The method converts RGB transmission into Optical Density (OD), where light absorption is linear:  A = -log10(I/I_0). 

2. Normalized Stain Vectors:
Each stain has a characteristic absorption spectrum across the Red, Green, and Blue channels. By taking a sample region containing only a single stain (e.g., pure Hematoxylin), one determines its unit-length absorption vector in OD space: 
v = [A_R,
     A_G,
     A_B] 
For a multi-stain slide, three stain vectors form a 3x3 stain matrix 
M = [ v_1R v_2R v_3R
      v_1G v_2G v_3G
      v_1B v_2B v_3B
    ]

(If only two stains are present, a dummy complementary orthogonal vector is constructed to complete the matrix.)

3. Matrix Inversion for Unmixing:
The measured OD vector at any pixel y = [A_R, A_G, A_B]^T is a linear combination of the unknown concentrations c = [c_1, c_2, c_3]^T of the individual dyes: 

y = Mc

By multiplying by the inverse matrix M^-1, the algorithm deconvolves the image into separate concentration maps for each stain:
c = M^-1 y

Primary ApplicationsQuantification: Accurately measures biomarker expression (e.g., Ki-67 or HER2 IHC scores) without interference from nuclear counterstains.  Stain Normalization: Acts as a pre-processing step for computer vision and deep learning pipelines to standardize variations in tissue staining across different labs.Separation Implementations: Widely available in open-source tools like ImageJ/Fiji (Colour Deconvolution plugin) and Python (scikit-image's rgb2hed / rgb2hdab modules). 