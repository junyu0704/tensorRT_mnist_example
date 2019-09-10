# tensorRT_mnist_example
在Jetson Nano使用TensorRT以MNIST為例

## NVIDIA EXAMPLE 
此範例參考NVIDIA範例再修改，

範例在/usr/src/tensorrt/samples/python


## 本例子可以學到什麼?

 * 1.在Jetson Nano上面訓練和推論
 * 2.如何使用TensorRT
 * 3.如何使用Plugin
 
## Jetson Nano環境

* 64GB記憶卡

* SWAP 8G
* Pillow == 5.1.0
* pycuda == 2019.1.2
* numpy == 1.16.0
* tensorflow-gpu == 1.13.1+nv19.3
* tensorrt == 5.0.6.3
* graphsurgeon == 0.3.2
* uff == 0.5.5
* cmake == 3.10.2

## Running the sample
>ref:https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#uff_custom_plugin

1. ### Download TensorRT_mnist_example
	```bash
	git clone https://github.com/junyu0704/tensorRT_mnist_example.git
	```
2. ### Build the plugin and its corresponding Python bindings.
	```bash
	mkdir build && pushd build
	cmake ..
	```

	>Note: If any of the dependencies are not installed in their default locations, you can manually specify them. For example:
	```bash  
	cmake .. \
		-DPYBIND11_DIR=/usr/local/pybind11/ \
		-DCUDA_ROOT=/usr/local/cuda-9.2/ \
		-DPYTHON3_INC_DIR=/usr/include/python3.6/ \
		-DNVINFER_LIB=/path/to/libnvinfer.so \
		-DTRT_INC_DIR=/path/to/tensorrt/include/
	```
	>cmake ..displays a complete list of configurable variables. If a variable is set to VARIABLE_NAME-NOTFOUND, then you’ll need to specify it manually or set the variable it is derived from correctly.

3. ### Build the plugin.
	```bash  
	make -j
	popd
	```
4. ### Run the sample to train the model: 
	```bash  
	python3 mnist.py
	```
5. ### Run inference using TensorRT with the custom clip plugin implementation: 
	```bash
	python3 mnist_pred.py
	```
6. ### Verify that the sample ran successfully. If the sample runs successfully you should see a match between the test case and the prediction.
	```bash
	=== Testing ===
	Loading Test Case: 1
	Prediction: 1
	```
