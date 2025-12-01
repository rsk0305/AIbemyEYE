import numpy as np



# -------------------------
# Utility / Data generator (simple)
# -------------------------
def example_scene_generator(num_sensors=16, duration_sec=1.0, rates=[200],
                            prob_1word=0.5, prob_2word=0.25, prob_bits=0.25, seed=None):
    """
    Produce a scene list. For 2word, create two node entries with meta['raw32'].
    All sensors generated at rate=200 Hz (unified rate).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    sensors = []
    idx = 0
    for i in range(num_sensors):
        rate = int(np.random.choice(rates))
        T = int(max(8, round(duration_sec * rate)))
        p = np.random.rand()
        if p < prob_1word:
            t = np.linspace(0, duration_sec, T)
            freq = np.random.uniform(0.2, 5.0)
            sig = 2000*np.sin(2*np.pi*freq*t + np.random.rand()*2*np.pi)
            sig += 10000 + np.random.normal(0,50,T)
            sensors.append({'id': idx, 'type':'1word', 'raw_rate':rate, 'raw':sig.astype(np.float32), 'meta':{}})
            idx += 1
        elif p < prob_1word + prob_2word:
            base = np.random.randint(0, 1<<20)
            noise = np.cumsum(np.random.normal(0, 100, T))
            raw32 = (base + noise).astype(np.int64)
            raw32 = np.clip(raw32, 0, (1<<32)-1)
            lsb = (raw32 & 0xFFFF).astype(np.int64)
            msb = ((raw32 >> 16) & 0xFFFF).astype(np.int64)
            sensors.append({'id': idx, 'type':'2word_lsb', 'raw_rate':rate, 'raw':lsb.astype(np.float32),
                            'meta': {'pair': idx+1, 'raw32': raw32}})
            idx += 1
            sensors.append({'id': idx, 'type':'2word_msb', 'raw_rate':rate, 'raw':msb.astype(np.float32),
                            'meta': {'pair': idx-1, 'raw32': raw32}})
            idx += 1
        else:
            bit_width = np.random.randint(1, 8)
            start = np.random.randint(0, 16-bit_width)
            if np.random.rand() < 0.6:
                values = np.cumsum(np.random.choice([0,1], size=T, p=[0.98,0.02])).astype(np.int64)
            else:
                values = (np.random.rand(T) < 0.02).astype(np.int64)
            word = (values & ((1<<bit_width)-1)) << start
            sensors.append({'id': idx, 'type':'bits', 'raw_rate':rate, 'raw':word.astype(np.float32),
                            'meta': {'bit_start':int(start), 'bit_end':int(start+bit_width-1)}})
            idx += 1
    return sensors

