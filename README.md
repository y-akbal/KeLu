A brand new activation function: KeLü (Keen Learning Unit):

It has continuous derivative at 0, while its third derivative has singularities at -3.5 and 3.5. Furthermore, compared to GELU it decays zero a bit faster. Both Flux and Jax implementations are included. For comparison we implement three well known networks. 

<p align="center">

<img src="assets/comparison.png" width="384" class="left"/>

<img src="assets/comparison_derivatives.png" width="384" class="center"/>
<img src="assets/comparison_second_derivatives.png" width="384" class="right"/>
</p>

Both Jax and Flux directories include implementation of the papers with respective abbreviations:

<li> Patches are all you need --- P</li>
<li> GPT2 --- GPT2 </li>

All the above models are trained with standard augmentation techniques (See SdP Repo).
</p>

# CIFAR100 (P)

| #Act.  |  Depth  | Patch_size | Kernel_size| Embed_Dim | Acc    | Loss     | 
| :---:  |  :-----:| :------:   | :------:   | :-----:   | :-----:| :-----:  | 
|  Relu  |  8      |  2         |     5      | 384       | 77.79  |  1.075   | 
|  Gelu  |  8      |  2         |     5      | 384       | 78.04  |  1.083   | 
|  Swish |  8      |  2         |     5      | 384       | 78.26  |  1.052   | 
|  KeLu  |  8      |  2         |     5      | 384       | 78.53  |  1.043   | 
|  KeLu  | 12      |  2         |     5      | 384       | 79.63  |  0.9787  | 
|  Gelu  | 12      |  2         |     5      | 384       | 79.14  |  0.9995  | 

# CIFAR10 (P)

| #Act.  |  Depth   | Patch_size | Kernel_size| Embed_Dim | Acc    | Loss     | 
| :---:  |  :-----: | :------:   | :------:   | :-----:   | :-----:| :-----:  | 
|  Relu  |  8       |  2         |     5      | 256       | 93.16  |  0.4382  | 
|  Gelu  |  8       |  2         |     5      | 256       | 93.23  |  0.4281  | 
|  KeLu  |  8       |  2         |     5      | 256       | 93.44  |  0.4274  | 

Note: For 150 epoch training, I am not able to reproduce the aforementioned results in "Patches are all you need article" for CIFAR10.
This is probably due to penalization methods.

# XXS GPT2 (Character Based- Being trained on 100MB Text of Newspaper articles)

|  #Params  | Embed_Dim| #Heads   |  #Blocks  |  KeLu - Val. Loss | gelu - Val. Loss| 
| :-------: | :-----:  | :------: | :------:  |  :-----:          |   :-----:       | 
|  55M      | 384      |  6       |  10       |                   |                 | 


