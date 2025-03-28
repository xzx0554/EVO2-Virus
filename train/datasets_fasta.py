class DNASequenceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256, stride=1, cache_capacity=10):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.file_path = file_path
        self.cache = OrderedDict()
        self.cache_capacity = cache_capacity

        index_file = file_path + '.index'
        if self._is_index_valid(index_file):
            try:
                with open(index_file, 'rb') as f:
                    self.sequence_info, self.cumulative_samples = pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                print(f"索引文件 {index_file} 已损坏，重新生成...")
                self._build_index(index_file)
        else:
            self._build_index(index_file)

    def _is_index_valid(self, index_file):
        if not os.path.exists(index_file):
            return False
        return os.path.getsize(index_file) > 0

    def _build_index(self, index_file):
        self.sequence_info = []
        self.cumulative_samples = [0]
        
        with open(self.file_path, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            pos = 0
            file_size = mm.size()

            while pos < file_size:
                header_pos = mm.find(b'>', pos)
                if header_pos == -1:
                    break

                header_end = mm.find(b'\n', header_pos)
                if header_end == -1:
                    seq_start = header_pos + 1
                else:
                    seq_start = header_end + 1

                next_header = mm.find(b'>', header_pos + 1)
                seq_end = next_header if next_header != -1 else file_size

                if seq_start >= seq_end:
                    pos = header_pos + 1
                    continue

                try:
                    seq_data = mm[seq_start:seq_end].replace(b'\n', b'').decode()
                    tokens = self.tokenizer(seq_data)
                    token_count = len(tokens)
                except UnicodeDecodeError:
                    pos = header_pos + 1
                    continue

                if token_count >= self.max_length:
                    num_samples = (token_count - self.max_length) // self.stride + 1
                elif token_count > 0:
                    num_samples = 1
                else:
                    num_samples = 0

                if num_samples > 0:
                    self.sequence_info.append((seq_start, seq_end))
                    self.cumulative_samples.append(self.cumulative_samples[-1] + num_samples)

                pos = max(seq_end, header_pos + 1)

            mm.close()

            temp_index = index_file + '.tmp'
            with open(temp_index, 'wb') as f:
                pickle.dump((self.sequence_info, self.cumulative_samples), f)
            os.replace(temp_index, index_file)  

    def __len__(self):
        return self.cumulative_samples[-1]

    def __getitem__(self, idx):
        seq_idx = bisect.bisect_right(self.cumulative_samples, idx) - 1
        local_idx = idx - self.cumulative_samples[seq_idx]
        seq_start, seq_end = self.sequence_info[seq_idx]

        if seq_idx in self.cache:
            tokens = self.cache[seq_idx]
            self.cache.move_to_end(seq_idx)
        else:
            with open(self.file_path, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                seq_data = mm[seq_start:seq_end].replace(b'\n', b'')
                seq = seq_data.decode('utf-8')
                mm.close()
            
            tokens = self.tokenizer(seq)
            
            if len(self.cache) >= self.cache_capacity:
                self.cache.popitem(last=False)
            self.cache[seq_idx] = tokens

        # 滑动窗口处理
        token_count = len(tokens)
        max_len = self.max_length
        stride = self.stride

        if token_count >= max_len:
            start = local_idx * stride
            end = start + max_len
            # 边界检查
            if end > token_count:
                end = token_count
                start = end - max_len
            input_tokens = tokens[start:end]
            label_tokens = tokens[start+1:end+1]
        else:
            input_tokens = tokens[:]
            label_tokens = tokens[1:]

        # 填充处理
        pad_id = 1
        eos_id = 0
        
        # 输入处理
        if len(input_tokens) < max_len:
            input_tokens = input_tokens + [eos_id] + [pad_id] * (max_len - len(input_tokens) - 1)
        else:
            input_tokens[-1] = eos_id

        # 标签处理
        if len(label_tokens) < max_len:
            label_tokens = label_tokens + [eos_id] + [pad_id] * (max_len - len(label_tokens) - 1)
        else:
            label_tokens[-1] = eos_id

        return input_tokens[:max_len], label_tokens[:max_len]
