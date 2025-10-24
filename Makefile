# -----------------------------------
# Mamba FPGA Inference Pipeline Makefile
# -----------------------------------

SRC := module/*.sv tb/*.sv
OUT := build/mamba_test
VCD := waveform/mamba_wave.vcd

all: $(OUT)
	@echo "🟢 Running simulation..."
	vvp $(OUT)
	@echo "📈 Launching GTKWave..."
	gtkwave $(VCD) &

mnist: build/mnist_test
	@echo "🧪 Simulating MNIST classifier..."
	vvp build/mnist_test
	gtkwave waveform/mnist_predict.vcd &

build/mnist_test: tb/tb_mnist_predict.sv $(SRC)
	@mkdir -p build waveform
	@echo "🔧 Building MNIST top-level..."
	iverilog -g2012 -o build/mnist_test tb/tb_mnist_predict.sv $(SRC)

$(OUT): $(SRC)
	@mkdir -p build waveform
	@echo "🛠️  Compiling Verilog sources..."
	iverilog -g2012 -o $(OUT) $(SRC)

clean:
	@rm -rf build waveform/*.vcd output.txt
	@echo "🧹 Cleaned build and waveform files."

synth:
	@echo "🚧 Starting synthesis (Yosys)..."
	yosys -p 'read_verilog module/*.sv; synth -top mamba_wrapper -json build/mamba.json'

.PHONY: all mnist clean synth
