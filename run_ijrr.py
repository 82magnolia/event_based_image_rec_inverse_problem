import os
import argparse
from utils.utils_viz import save_image
from utils.utils_load import IJRRDataloader
from utils.utils_img_rec import ImageReconstructor
from ast import literal_eval
from tqdm import tqdm


def arg_parser():
    parser = argparse.ArgumentParser(description='Split a big events.txt files to small chunks(1s).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', type=str, help='Choose a sequence name from IJRR dataset.')
    parser.add_argument('--num_events', type=int, default=30000, help="Number of events to load")
    parser.add_argument('--output_path', type=str, default="log/", help="Where to save the output images")
    parser.add_argument('--cost', type=str, default='l2', help="Type of cost function to use")
    return parser


def image_reconstruction(args):
    dataset_path = os.path.join("data", args.dataset)
    assert os.path.isdir(dataset_path), "The dataset doesn't exist!!!"

    img_file_path = os.path.join("data", args.dataset, "images.txt")
    raw_img_timestamp = open(img_file_path, 'r').readlines()
    img_timestamps = {}
    for timestamp_idx in range(len(raw_img_timestamp)):
        raw_timestamp = raw_img_timestamp[timestamp_idx]
        timestamp = literal_eval(raw_timestamp.strip().split(' ')[0])
        img_name = raw_timestamp.strip().split(' ')[-1].split('/')[-1]
        img_timestamps[img_name] = timestamp

    full_keys = list(img_timestamps.keys())
    for img_name in tqdm(full_keys):
        img_timestamp = img_timestamps[img_name]
        num_events = args.num_events
        output_path = args.output_path
        ijrr_loader = IJRRDataloader(dataset_path)
        events_torch, flow_torch = ijrr_loader.load_events_and_flow(img_timestamp, num_events)
        image_reconstructor = ImageReconstructor(flow_torch)
        
        if args.cost == 'l1':
            img_rec = image_reconstructor.image_rec_from_events_l1(events_torch, reg_weight=1e-1)
        else:
            img_rec = image_reconstructor.image_rec_from_events_l2(events_torch, reg_weight=3e-1)
        # img_rec_denoiser = image_reconstructor.image_rec_from_events_cnn(events_torch, weight1=2.5, weight2=1.3)
        # Save images
        save_image(img_rec, output_path, img_name)
    # save_image(img_rec_denoiser, output_path, "denoiser.png")

if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    image_reconstruction(args)
