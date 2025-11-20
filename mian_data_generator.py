import numpy as np

def generate_multimodal_data(
    num_sensors=100,
    seq_len=200,
    prob_1word=0.4,
    prob_2word=0.3,
    prob_bits=0.3,
    seed=None
):
    """
    Generate randomized multimodal sensor data with labels and metadata.

    Returns:
        data: (num_sensors, seq_len) ndarray of int (0~65535)
        labels: list of sensor-type labels
            1 → 1word (16bit continuous)
            2 → 2word (MSB/LSB pair)
            3 → bits (partial bitfield)
        meta: list of metadata dictionaries per sensor
    """

    if seed is not None:
        np.random.seed(seed)

    data = np.zeros((num_sensors, seq_len), dtype=np.uint16)
    labels = []
    meta = []

    # Random type assignment
    sensor_types = np.random.choice(
        [1, 2, 3],
        size=num_sensors,
        p=[prob_1word, prob_2word, prob_bits]
    )

    # To track MSB-LSB pairing for 2-word sensors
    used_indices = set()

    for i in range(num_sensors):
        t = sensor_types[i]

        # -----------------------------------------
        # 1) 1WORD SENSOR (continuous 16bit signal)
        # -----------------------------------------
        if t == 1:
            base = np.random.uniform(0, 1)
            noise = np.random.normal(0, 0.03, seq_len)
            signal = base + np.cumsum(noise)  # random walk
            signal = np.clip(signal * 20000, 0, 65535)  # scale to 16bit
            data[i] = signal.astype(np.uint16)

            labels.append(1)
            meta.append({
                "type": "1word",
                "bit_start": 0,
                "bit_end": 15
            })

        # -----------------------------------------
        # 2) 2WORD SENSOR (32bit: MSB/LSB pairing)
        # -----------------------------------------
        elif t == 2:

            # If already used as MSB or LSB, skip (meta is created earlier)
            if i in used_indices:
                continue

            # Ensure valid index for MSB-LSB pair
            if i < num_sensors - 1:
                lsb_idx = i
                msb_idx = i + 1

                used_indices.add(lsb_idx)
                used_indices.add(msb_idx)

                labels.extend([2, 2])  # two rows represent one 32-bit sensor

                meta.append({
                    "type": "2word",
                    "role": "LSB",
                    "pair": msb_idx
                })
                meta.append({
                    "type": "2word",
                    "role": "MSB",
                    "pair": lsb_idx
                })

                # Generate continuous 32bit increasing signal
                base = np.random.uniform(0, 10)
                noise = np.random.normal(0, 0.5, seq_len)
                signal32 = base + np.cumsum(noise)
                signal32 = np.clip(signal32 * 5000, 0, (1 << 32) - 1).astype(np.uint32)

                # Split into LSB(16bit), MSB(16bit)
                lsb = signal32 & 0xFFFF
                msb = (signal32 >> 16) & 0xFFFF

                data[lsb_idx] = lsb.astype(np.uint16)
                data[msb_idx] = msb.astype(np.uint16)

            else:
                # If last index, fallback → treat as 1word
                labels.append(1)
                base = np.random.uniform(0, 1)
                noise = np.random.normal(0, 0.03, seq_len)
                signal = base + np.cumsum(noise)
                signal = np.clip(signal * 20000, 0, 65535)
                data[i] = signal.astype(np.uint16)
                meta.append({
                    "type": "fallback_1word_due_to_boundary"
                })

        # -----------------------------------------
        # 3) BITFIELD SENSOR (1~15bits)
        # -----------------------------------------
        elif t == 3:
            bit_width = np.random.randint(1, 16)  # 1~15 bits
            start_bit = np.random.randint(0, 16 - bit_width)
            end_bit = start_bit + bit_width - 1

            # Generate random toggle/control signal
            toggles = np.random.choice([0, 1], size=seq_len, p=[0.9, 0.1])
            value = toggles.astype(np.uint16) << start_bit

            data[i] = value

            labels.append(3)
            meta.append({
                "type": "bits",
                "bit_start": start_bit,
                "bit_end": end_bit,
                "bit_width": bit_width
            })

    return data, labels, meta


# =======================
# Example Execution
# =======================
if __name__ == "__main__":
    data, labels, meta = generate_multimodal_data(
        num_sensors=20,
        seq_len=50,
        seed=42
    )

    print("Data shape:", data.shape)
    for i, m in enumerate(meta[:10]):
        print(f"{i} → label={labels[i]}, meta={m}")

'''
Output
1 = 1word
2 = 2word (MSB/LSB pair)
3 = bits


{
  'type': '2word',
  'role': 'LSB',
  'pair': 5
}
{
  'type': 'bits',
  'bit_start': 3,
  'bit_end': 7,
  'bit_width': 5
}
{
  'type': '1word',
  'bit_start': 0,
  'bit_end': 15
}
'''
