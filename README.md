# 全方位地震动反应谱生成

## 安装
`pip install omnispectrum`

## 例程
### spectrum
```python
import omnispectrum
# acc加速度时程，dt，是否进行基线修正
spect = omnispectrum.spectrum(ew, t, False)
# 计算反应谱，start，end，step(开始周期，结束周期，步长)
spect.get_sa(0, 2, 0.1)
print(spect.sa)
```

### omnispectrum

```python
import omnispectrum
# csv为三向地震动，格式参照实例文件
omni = omnispectrum.OmniSpectrum('022DLB.csv')
omni.any_angle_spectrum()
```

### 批量转换生成word报告
```python
import omnispectrum
csv2docx(path, output, baseline_correction=False)
```

### 作为模块使用
```shell
python -m omnispectrum.omnispectrum --i path
python -m omnispectrum.omnispectrum --help
```

### Example
[例程](docs/强震动反应谱分析.ipynb)