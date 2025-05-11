## License

​	This code is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.



## Usage 

To reproduce Figure 1, execute the following command in the terminal:

```
python numerical.py exphyperopt
```
and then run `plotHyperSearch.ipynb`.



To reproduce Figure 2.a and Figure 2.b, execute the following commands in the terminal:

```
python numerical.py exp1
python numerical.py exp2
```
and then run `plotNumerical.ipynb`.



To reproduce Figure 3, Table 3, and Table 4, first ensure that `DeepSpeed-Chat` has been set up (https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md) on the device, and then execute the following commands in the terminal:

```
python empirical.py rmstaticpythia70mlastsplr1em5wd1em1ep1
```
and then run `plotEmpirical.ipynb`. To change the dataset or the base model, edit the corresponding settings in the file `empirical`. The `main.py` file is modified based on the original file of `DeepSpeed-Chat`. 



To reproduce Figure 4, first set up  `DeepSpeed-Chat`, then execute

```
python run_trans.py shpllama1b
```

and then run `plotLast.ipynb`. 

## Citation
Please cite our paper if you find this work useful:

```bibtex
@inproceedings{yao2025sparse,
  title     = {Leveraging Sparsity for Sample-Efficient Preference Learning: A Theoretical Perspective},
  author    = {Yunzhen Yao, Lie He, Michael Gastpar},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```
