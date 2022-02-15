# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import numpy as np
import sys
from pathlib import Path
import tensorflow as tf
from deepfill_ops import init_inpaint_network, get_gpu_list

# Import classes from deepformable library
root_path = Path(__file__).parent.resolve()
sys.path.insert(0, str(root_path.parent / "deepformable/utils"))
from inpaint_utils import NoInpaint, generate_marker_mask

class DeepfillInpaint(NoInpaint):
    def __init__(
        self,
        processing_workers=1,
        writing_workers=8,
        max_task_size=32,
        max_write_size=24,
    ):
        config=tf.ConfigProto()
        tf.reset_default_graph()
        self.sess = tf.Session(config=config)
        self.gpu_list = get_gpu_list()
        self.inpaint_inputs, self.inpaint_outputs = init_inpaint_network(self.sess, self.gpu_list)
        super().__init__(
            processing_workers, writing_workers, max_task_size,
            max_write_size, use_multiprocessing=False)
    
    def process_data(self, data):
        feed_dict = {}
        for i, (_, undistorted_frame, cur_annotations, markers_world, mtx) in enumerate(data):
            mask = generate_marker_mask(undistorted_frame, cur_annotations, markers_world, mtx)
            feed_dict[self.inpaint_inputs[i]] = (np.expand_dims(undistorted_frame, 0), np.expand_dims(mask, 0))
        net_out = self.sess.run(self.inpaint_outputs[:len(feed_dict)], feed_dict=feed_dict)
        return [(p[0], img[0][...,[2,1,0]]) for p, img in zip(data, net_out)]
        # return [(p[0], p[1]) for p in data]
    
    # Modified this for multi-gpu batched input
    def processing_worker(self):
        worker_exit = False
        while not worker_exit:
            data = []
            for _ in self.gpu_list:
                cur_task = self.tasks.get()
                if cur_task is None:
                    worker_exit = True
                    break
                else:
                    data.append(cur_task)
            for result in self.process_data(data):
                self.results.put(result)
                self.tasks.task_done()
        self.tasks.task_done()