# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import numpy as np
import cv2
import multiprocessing, queue
import threading

class WorkerBase:
    def __init__(
        self,
        processing_workers=1,
        writing_workers=1,
        max_task_size=32,
        max_write_size=16,
        use_multiprocessing=False,
    ):
        self.use_multiprocessing = use_multiprocessing
        if use_multiprocessing:
            self.tasks = multiprocessing.JoinableQueue(maxsize=max_task_size)
            self.results = multiprocessing.JoinableQueue(maxsize=max_write_size)
            self.processing_workers = [
                multiprocessing.Process(target=self.processing_worker) for _ in range(processing_workers)]
            self.writing_workers = [
                multiprocessing.Process(target=self.writing_worker) for _ in range(writing_workers)]
        else:
            self.tasks = queue.Queue(maxsize=max_task_size)
            self.results = queue.Queue(maxsize=max_write_size)
            self.processing_workers = [
                threading.Thread(target=self.processing_worker) for _ in range(processing_workers)]
            self.writing_workers = [
                threading.Thread(target=self.writing_worker) for _ in range(writing_workers)]
        for p in [*self.processing_workers, *self.writing_workers]:
            p.start()
    
    def wait_finish(self):
        for _ in self.processing_workers:
            self.tasks.put(None)
        self.tasks.join()
        for _ in self.writing_workers:
            self.results.put(None)
        self.results.join()
        for p in [*self.processing_workers, *self.writing_workers]:
            p.join()  
    
    def processing_worker(self):
        while True:
            data = self.tasks.get()
            if data is None:
                self.tasks.task_done()
                break
            self.results.put(self.process_data(data))
            self.tasks.task_done()
    
    def writing_worker(self):
        while True:
            result = self.results.get()
            if result is None:
                self.results.task_done()
                break
            self.write_result(result)
            self.results.task_done()
    
    def process_data(self, data):
        return data

    def write_result(self, result):
        print(result)

    def __call__(self, data):
        if len(self.processing_workers) == 0:
            if len(self.writing_workers) == 0:
                self.write_result(self.process_data(data))
            else:
                self.results.put(self.process_data(data))
            return
        self.tasks.put(data)


class NoInpaint(WorkerBase):
    def __init__(
        self,
        processing_workers=0,
        writing_workers=8,
        max_task_size=24,
        max_write_size=24,
        use_multiprocessing=False,
    ):
        super().__init__(
            processing_workers, writing_workers, max_task_size,
            max_write_size, use_multiprocessing)

    def process_data(self, data):
        file_path, undistorted_frame, _, _, _ = data
        return file_path, undistorted_frame
    
    def write_result(self, result):
        cv2.imwrite(*result)


def generate_marker_mask(
    undistorted_frame, 
    cur_annotations, 
    markers_world, 
    mtx,
    margin_ratio=10,
):
    mask = np.zeros(undistorted_frame.shape)
    for ann, markers in zip(cur_annotations, markers_world): 
        markersw_margin = (markers - np.roll(markers, 2, 1))/margin_ratio + markers
        markersw_margin = cv2.projectPoints(
            markersw_margin.reshape(-1,3), 
            np.array(ann['rvec']), np.array(ann['tvec']),
                mtx, None)[0].reshape(-1, 4, 2)
        for p in markersw_margin:
            cv2.fillConvexPoly(mask, np.int32(p), (1.0, 1.0, 1.0), cv2.LINE_4)
    return mask


class OpenCVInpaint(NoInpaint):
    def __init__(
        self,
        processing_workers=8,
        writing_workers=8,
        max_task_size=24,
        max_write_size=24,
        use_multiprocessing=False,
    ):
        super().__init__(
            processing_workers, writing_workers, max_task_size,
            max_write_size, use_multiprocessing)

    def process_data(self, data):
        file_path, undistorted_frame, cur_annotations, markers_world, mtx = data
        mask = generate_marker_mask(undistorted_frame, cur_annotations, markers_world, mtx)
        inpainted_frame = cv2.inpaint(
            np.uint8(undistorted_frame), 
            np.uint8(mask[...,0]*255), 
            5, cv2.INPAINT_TELEA)
        return file_path, inpainted_frame

if __name__ == '__main__':
    test_worker = WorkerBase(
        processing_workers=4, use_multiprocessing=True)
    for i in range(10, 23):
        # print(i)
        test_worker(i)
    test_worker.wait_finish()