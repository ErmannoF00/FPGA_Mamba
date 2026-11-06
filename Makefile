PYTHON ?= python3
VERILATOR ?= verilator
IVERILOG ?= iverilog
VVP ?= vvp
YOSYS ?= yosys
VIVADO ?= vivado
SEEDS ?= 1729 1730 1731
CYCLES ?= 2400
LANES ?= 4

.PHONY: test reference directed verilator iverilog synth vivado lint clean smoke

# The default CI gate is verilator, not a stub or a Python-only check.
test: reference lint directed verilator iverilog

reference:
	$(PYTHON) -m unittest tests.test_fixed_point -v

lint:
	$(VERILATOR) --lint-only -Wall --top-module selective_scan rtl/selective_scan.sv

directed:
	@mkdir -p build
	$(IVERILOG) -g2012 -s tb_directed -Wall -o build/direct_tb rtl/selective_scan.sv tb/tb_directed.sv
	$(VVP) build/direct_tb

verilator:
	@mkdir -p build
	$(VERILATOR) --binary --timing -j 2 \
		--top-module tb_selective_scan -GLANES=$(LANES) \
		--Mdir build/vlt_$(LANES) -o scan_tb \
		rtl/selective_scan.sv tb/tb_selective_scan.sv
	@set -e; for seed in $(SEEDS); do \
	  $(PYTHON) python/generate_vectors.py --lanes $(LANES) --cycles $(CYCLES) --seed $$seed --output build/vectors_$(LANES)_$$seed.txt; \
	  build/vlt_$(LANES)/scan_tb +VECTORS=build/vectors_$(LANES)_$$seed.txt; \
	done

iverilog:
	@mkdir -p build
	$(IVERILOG) -g2012 -s tb_selective_scan -Ptb_selective_scan.LANES=$(LANES) -Wall \
		-o build/ivl_tb rtl/selective_scan.sv tb/tb_selective_scan.sv
	@set -e; for seed in $(SEEDS); do \
	  $(PYTHON) python/generate_vectors.py --lanes $(LANES) --cycles $(CYCLES) --seed $$seed --output build/vectors_$(LANES)_$$seed.txt; \
	  $(VVP) build/ivl_tb +VECTORS=build/vectors_$(LANES)_$$seed.txt; \
	done

synth:
	@mkdir -p build
	$(YOSYS) -l build/yosys.log -p 'read_verilog -sv rtl/selective_scan.sv; hierarchy -top selective_scan; proc; opt; check -assert; synth -top selective_scan; stat'

# Supply the actual device part; synthesis/P&R are not the same as board bring-up.
# Example: make vivado PART=xc7a35tcpg236-1 CLK_NS=10
vivado:
	@test -n "$(PART)" || (echo 'ERROR: provide PART=<your FPGA part>' >&2; exit 2)
	mkdir -p build/vivado
	PART="$(PART)" CLK_NS="$(or $(CLK_NS),10)" $(VIVADO) -mode batch \
		-source scripts/vivado_synth.tcl -log build/vivado/run.log -journal build/vivado/run.jou

smoke: reference

clean:
	rm -rf build
