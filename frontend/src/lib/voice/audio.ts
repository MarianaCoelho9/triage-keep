export const convertFloat32ToInt16 = (buffer: Float32Array): ArrayBuffer => {
  let length = buffer.length;
  const output = new Int16Array(length);
  while (length--) {
    output[length] = Math.min(1, Math.max(-1, buffer[length])) * 0x7fff;
  }
  return output.buffer;
};

export const downsampleBuffer = (
  buffer: Float32Array,
  sampleRate: number,
  outSampleRate: number,
): Float32Array => {
  if (outSampleRate === sampleRate) return buffer;
  if (outSampleRate > sampleRate) {
    throw new Error("Downsampling rate should be smaller than the original sample rate.");
  }

  const sampleRateRatio = sampleRate / outSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;
    for (let index = offsetBuffer; index < nextOffsetBuffer && index < buffer.length; index += 1) {
      accum += buffer[index];
      count += 1;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }

  return result;
};
