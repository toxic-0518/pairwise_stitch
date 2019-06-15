# Pairwise Stitching
A python implementation for multi-page tiff stitching.

## Prerequisites

### Environment
```yaml
Python >= 3.6
OpenCV (3.3 <= version <= 3.4.16) ( with opencv-contrib)
numpy
```

For your convenience, I recommend you use anaconda to build enviromnment.

miniconda documentation: [link](https://docs.conda.io/en/latest/miniconda.html)

### Test Data (Optional)
link: [`https://pan.baidu.com/s/1X6-DxoKUwHy9y3ZHMdNfHQ`](https://pan.baidu.com/s/1X6-DxoKUwHy9y3ZHMdNfHQ).  access code: `pbtd` 

## Useage

```sh
git clone https://github.com/toxic-0518/pairwise_stitch.git
cd pairwise_stitch
conda env create -f requirements.yml
conda activate pairwise_stitch
```

To test code, you should download tiff files or use your own tiff files. Put it into right place,

then update code in `pairwise_stitch.py`.

```python
if __name__ == '__main__':
    """your test code goes here"""
    pairwise_stitch(your_tiff_filepath_1, your_tiff_filepath_2, debug=True)
```

then run
```bash
python pairwise_stitch.py
```


### Some Examples

SIFT matching result:

![image](https://github.com/toxic-0518/pairwise_stitch/blob/master/images/1.png)
![image](https://github.com/toxic-0518/pairwise_stitch/blob/master/images/2.png)
![image](https://github.com/toxic-0518/pairwise_stitch/blob/master/images/3.png)
![image](https://github.com/toxic-0518/pairwise_stitch/blob/master/images/4.png)

stitch result:

![image](https://github.com/toxic-0518/pairwise_stitch/blob/master/images/result_1.png)
![image](https://github.com/toxic-0518/pairwise_stitch/blob/master/images/result_2.png)
![image](https://github.com/toxic-0518/pairwise_stitch/blob/master/images/result_3.png)
![image](https://github.com/toxic-0518/pairwise_stitch/blob/master/images/result_4.png)



