# attn/atari — Paper Tape Is All You Need

A single-layer transformer with self-attention, trained and run entirely on an
Atari 8-bit computer (800). Written in C-89 and compiled with cc65 targeting
the 6502 processor at 1.79 MHz.

The network learns to reverse 8-digit sequences of random digits [0–9], using
fixed-point arithmetic throughout — no floating point anywhere.

The Atari code was constructed by first porting the PDP-11 assembler (`ai.asm`) to
Mac C (clang). Once that was working correctly, the Mac C code was ported to cc65
and targeted the Atari 8-bit platform.

## Credits

This project is a port of two prior works:

**ATTN/11 — "Paper Tape Is All You Need"**  
Original transformer implementation in PDP-11 assembly by dbrll.  
https://github.com/dbrll/ATTN-11

**PDP-11 source port by Dave Plummer**  
Dave Plummer's PDP-11 assembly port, which this C implementation is derived
from and translates into portable C89.  
https://github.com/davepl/pdpsrc/tree/main/bsd/attn

## Network Architecture

| Parameter       | Value                              |
|-----------------|------------------------------------|
| Sequence length | 8                                  |
| Model dimension | 16                                 |
| Vocabulary size | 10                                 |
| Total weights   | 1,216                              |
| Attention heads | 1                                  |
| Task            | Sequence reversal of random digits |

Fixed-point formats:

| Type  | C type     | Scale       | Use                          |
|-------|------------|-------------|------------------------------|
| Q7.8  | `int16_t`  | / 256.0     | Weights, activations         |
| Q15   | `int16_t`  | / 32768.0   | Gradient accumulators        |
| Q16   | `int32_t`  | / 256.0     | Weight accumulators (32-bit) |
| Q12   | `int32_t`  | / 4096.0    | Loss table entries           |

## Files

| File              | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| `atari-ai.c`      | Full training implementation (forward, backward, SGD update, I/O)  |
| `atari-ai.h`      | Type definitions, struct layouts, constants, extern declarations   |
| `atari-build.sh`  | Build script — invokes cc65 toolchain, produces `atari-ai.xex`     |
| `mac-ai.c`        | Mac reference implementation (native C, float)                     |
| `mac-ai.h`        | Mac reference header                                               |
| `mac-build.sh`    | Mac build script                                                   |

## Building

The Atari code requires the [cc65](https://cc65.github.io/) toolchain 
installed and on `PATH`.

```bash
./atari-build.sh
```

Produces `atari-ai.xex` — a standard Atari executable binary.

## Running

### Emulator (atari800) — Mac

```bash
brew install atari800
atari800 -xl -turbo -nosound atari-ai.xex
```

`-turbo` removes the speed cap and runs at maximum host CPU speed. Training
completes in seconds rather than the ** 11 hours! ** it would take at real 1.79 MHz.

### Real Hardware

Copy `atari-ai.xex` to an SIO2SD, FujiNet, or similar device and load from
DOS or a loader. Training runs at native 1.79 MHz; each step produces one
line of output so you have continuous feedback.

## Implementation Notes

## Function names
- Generally are traceable back to AI.ASM (Plummer PDP-11 assembly)

### C89 Compliance

cc65 enforces strict C89. All variable declarations were hoisted to the top of
their enclosing block before any executable statements. 

### Fixed-Point Arithmetic

All math uses integer types. Multiplication of two Q8 values produces a Q16
result in `int32_t`; right-shift by 8 recovers Q8. The 32-bit accumulator
split into hi/lo `int16_t` pairs is a direct translation of the PDP-11
register-pair (R0:R1) used by the MUL instruction.

### Lookup Tables

- `exptbl[256]` — Q8 values of exp(−i/32) for i = 0..255, used by softmax
- `logtbl[257]` — Q12 values of −ln(x/256)×4096 for x = 0..256, used by cross-entropy loss

## Training

- Optimizer: SGD with per-group learning rate shifts (bit shifts in place of division)
- Loss: cross-entropy via softmax + log-table lookup
- In my testing, the model converged to 100% accuracy in about 275 steps.
- Kept 350 steps in code to be true to original.  Could lower to reduce training time.

## Sample Output

```
training...
 step   50 loss=0.0051 accuracy=0.252
 ...
 step  100 loss=0.0056 accuracy=0.340
 ...
 step  150 loss=0.0073 accuracy=0.435
 ...
 step  300 loss=0.0000 accuracy=0.982
 ...
 step  350 loss=0.0000 accuracy=1.000

test
 0 9 8 1 2 3 8 5 -> 5 8 3 2 1 8 9 0  OK
 8 1 6 1 6 7 8 7 -> 7 8 7 6 1 6 1 8  OK
 ...
 accuracy: 10/10
```