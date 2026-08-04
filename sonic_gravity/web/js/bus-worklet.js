/**
 * bus-worklet.js — the multiband bus, running on the audio thread.
 *
 * Thin wrapper: all the DSP lives in `buscomp.js`, which is testable in node
 * and checked sample-for-sample against `sonic_gravity/bus.py`. This file only
 * handles the AudioWorklet contract.
 */

import { DEFAULT_BANDS, DEFAULT_CROSSOVERS, MultibandBus } from "./buscomp.js";

// One message every ~50 ms is plenty for a meter; the audio thread must not
// spend its budget posting messages.
const REPORT_EVERY = 24; // blocks of 128 ≈ 64 ms at 48 kHz

class BusProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options?.processorOptions ?? {};
    this.bus = new MultibandBus(
      sampleRate,
      opts.bands ?? DEFAULT_BANDS,
      opts.crossovers ?? DEFAULT_CROSSOVERS,
      opts.channels ?? 2
    );
    this.enabled = opts.enabled !== false;
    this.count = 0;
    this.port.onmessage = (e) => {
      if (e.data?.type === "enable") this.enabled = !!e.data.value;
    };
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || input.length === 0) return true;

    const len = input[0].length;
    if (!this.enabled) {
      for (let c = 0; c < output.length; c++) {
        output[c].set(input[Math.min(c, input.length - 1)]);
      }
      return true;
    }

    // The bus detects on the mono sum and filters each channel with its own
    // state, so the stereo image survives.
    const chans = [];
    for (let c = 0; c < output.length; c++) chans.push(input[Math.min(c, input.length - 1)]);
    const gr = this.bus.process(chans, output, len);

    if (++this.count >= REPORT_EVERY) {
      this.count = 0;
      this.port.postMessage({ type: "gr", gr: Array.from(gr) });
    }
    return true;
  }
}

registerProcessor("multiband-bus", BusProcessor);
