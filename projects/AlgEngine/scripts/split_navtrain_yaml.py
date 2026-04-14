"""Split navtrain YAML into N chunks for memory-friendly evaluation."""
import argparse
import os
import yaml


def split_yaml(input_path, num_chunks, output_dir):
    with open(input_path, 'r') as f:
        data = yaml.safe_load(f)

    tokens = data.get('tokens') or data.get('scenario_tokens')
    log_names = data.get('log_names', [])
    assert tokens is not None, "No tokens or scenario_tokens found in YAML"

    # Build log_name -> tokens mapping
    # We split by log_names to keep scenarios from the same log together
    log_to_tokens = {}
    token_set = set(tokens)
    for t in tokens:
        # tokens don't directly map to log_names in the yaml,
        # so we just split tokens evenly
        pass

    chunk_size = (len(tokens) + num_chunks - 1) // num_chunks
    os.makedirs(output_dir, exist_ok=True)

    chunk_paths = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(tokens))
        if start >= len(tokens):
            break
        chunk_tokens = tokens[start:end]

        chunk_data = {
            '_convert_': data.get('_convert_', 'all'),
            '_target_': data.get('_target_', 'navsim.common.dataclasses.SceneFilter'),
            'frame_interval': data.get('frame_interval', 1),
            'has_route': data.get('has_route', True),
            'log_names': log_names,  # keep all log_names, filtering is by tokens
            'tokens': chunk_tokens,
        }

        chunk_name = f"navtrain_chunk_{i:03d}_of_{num_chunks:03d}.yaml"
        chunk_path = os.path.join(output_dir, chunk_name)
        with open(chunk_path, 'w') as f:
            yaml.dump(chunk_data, f, default_flow_style=False)

        chunk_paths.append(chunk_path)
        print(f"Chunk {i}: {len(chunk_tokens)} tokens -> {chunk_path}")

    print(f"\nTotal: {len(tokens)} tokens split into {len(chunk_paths)} chunks")
    return chunk_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_yaml', help='Path to navtrain YAML file')
    parser.add_argument('--num-chunks', type=int, default=10, help='Number of chunks')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same dir as input)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input_yaml), 'chunks')

    split_yaml(args.input_yaml, args.num_chunks, args.output_dir)
