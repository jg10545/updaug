# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import os

from updaug._loader import dataset_generator
from updaug._trainstep import _build_discriminator_training_step
from updaug._trainstep import _build_generator_training_step
from updaug._models import build_generator, build_discriminator
from updaug._loss import _l1_loss

def _get_example_images(testdata, testlabels, outputshape):
    df = pd.DataFrame({"filepath":testdata, "label":testlabels})
    num_domains = df.label.max()+1
    samples = []
    for d in range(num_domains):
        subset = df[df.label == d]
        img = Image.open(subset.filepath.values[0]).resize(outputshape)
        samples.append(np.array(img).astype(np.float32)/255)
    
    samples = np.stack(samples, 0)
    return samples


class Trainer(object):
    """
    """
    
    
    def __init__(self, logdir, traindata, trainlabels, 
                 testdata, testlabels, steps_per_epoch,
                 crop=False, flip=True, rot=False,
                 imshape=(128,128), filetype="png",
                 batch_size=64, num_parallel_calls=6,
                 lr=1e-3, lr_decay=0, 
                 lam1=1, lam2=10, lam3=10, lam4=100,
                 strategy=None):
        """
        :logdir: (string) path to log directory
        :traindata: list of strings; paths to each image in the dataset
        :trainlabels
        :testdata: list of strings; a batch of images to use for out-of-sample
            visualization in tensorboard
        :testlabels:
        :imshape: image shape to use
        :batch_size: batch size for traniing
        :num_parallel_calls: number of cores to use for loading/preprocessing images
        :lr: learning rate for Adam optimizer
        :lr_decay:
        :weights: list of 6 floats- weights to use for each perceptual loss
            component. See paper for details.
        :weightdecay: L2 loss applied to model weights
        """
        self.logdir = logdir
        self.imshape = imshape
        self.num_domains = np.max(trainlabels) + 1
        pairs_per_epoch = steps_per_epoch * batch_size
        
        # ------- SET UP WRITER TO LOG TENSORBOARD METRICS -------
        if logdir is not None:
            self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
            self._file_writer.set_as_default()
        self.step = 0
        # ------- SET UP STRATEGY -------
        if strategy is None:
            strategy = tf.distribute.get_strategy()
        self.strat = strategy
        # ------- INITIALIZE MODELS -------
        self.models = {}
        with self.strat.scope():
            self.models["generator"] = build_generator(self.num_domains)
            self.models["discriminator"] = build_discriminator(self.num_domains)
        # ------- INITIALIZE OPTIMIZER -------
        with self.strat.scope():
            if lr_decay > 0:
                lr = tf.keras.experimental.CosineDecayRestarts(lr, lr_decay,
                                                           t_mul=2., m_mul=1.,
                                                           alpha=0.)
            self.optimizer = tf.keras.optimizers.Adam(lr)
        
        # ------- SET UP DATASETS -------
        self.ds_gen = dataset_generator(traindata, trainlabels, pairs_per_epoch,
                                        num_parallel_calls=num_parallel_calls,
                                        outputshape=imshape, filetype=filetype,
                                        batch_size=batch_size, crop=crop, flip=flip,
                                        rot=rot, seed=False) 
        self.testds = dataset_generator(testdata, testlabels, pairs_per_epoch,
                                        num_parallel_calls=num_parallel_calls,
                                        outputshape=imshape, filetype=filetype,
                                        batch_size=batch_size, crop=crop, flip=False,
                                        rot=False, seed=1) 
        
        
        # ------- SET UP TRAINING STEP -------
        
        step_fn = _build_generator_training_step(self.models["generator"],
                                                 self.models["discriminator"],
                                                 self.optimizer, lam1, lam2, lam3, lam4)    
        @tf.function
        def training_step(x,y):
            per_example_losses = self.strategy.run(step_fn, args=(x,y))

            lossdict = {k:self.strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, 
                    per_example_losses[k], axis=None)
                    for k in per_example_losses}

            return lossdict
        self.trainstep = training_step
        
                # ------- GET SOME EXAMPLE IMAGES -------
        self._sample = _get_example_images(testdata, testlabels, imshape)
        
        self._visualize_outputs()

    
    def fit(self, epochs=1, save=True):
        """
        """
        for e in tqdm(range(epochs)):
            ds = next(self.ds_gen)
            ds = self.strategy.experimental_distribute_dataset(ds)
            for x0, y0, x1, y1 in self.ds:
                loss = self.trainstep(x0, y0, x1, y1)
                self._record_scalars(loss=loss)
                self.step += 1
            
            self.evaluate()
            if save:
                self.save()
            self._visualize_outputs()
            
    def evaluate(self):
        loss = []
        for x0, y0, x1, y1 in self.testds:
            fake0 = self.models["generator"](x0, y1)
            fake1 = self.models["generator"](x1, y0)
            recon0 = self.models["generator"](fake0, y0)
            recon1 = self.models["generator"](fake1, y1)
            
            recon_loss = 0.5*(_l1_loss(x0, recon0).numpy().mean() + \
                              _l1_loss(x1, recon1).numpy().mean())
            loss.append(recon_loss)
        self._record_scalars(test_recon_loss=np.mean(recon_loss))
        
    def save(self):
        """
        Write model(s) to disk
        
        """
        for m in self.models:
            path = os.path.join(self.logdir, m)
            self.models[m].save(path, overwrite=True, save_format="tf")
            
    def _visualize_outputs(self):
        img = [self._sample]
        N = self._sample.shape[0]
        for d in range(self.num_domains):
            output_domain = d*np.ones(N, dtype=np.int64)
            output_domain = tf.one_hot(output_domain, self.num_domains)
            img.append(self.models["generator"](self._sample, output_domain))
            
        img = np.concatenate([np.concatenate([i[j] for j in range(N)], 0)
               for i in img], 1)
        
        self._record_images(domain_reconstructions=img)
            
    def _record_scalars(self, metric=False, **scalars):
        for s in scalars:
            tf.summary.scalar(s, scalars[s], step=self.step)
            
            if metric:
                if hasattr(self, "_mlflow"):
                    self._log_metrics(scalars, step=self.step)
            
    def _record_images(self, **images):
        for i in images:
            tf.summary.image(i, images[i], step=self.step, max_outputs=10)
            
    def _record_hists(self, **hists):
        for h in hists:
            tf.summary.histogram(h, hists[h], step=self.step)
            
        
    def _get_current_learning_rate(self):
        # return the current value of the learning rate
        # CONSTANT LR CASE
        if isinstance(self._optimizer.lr, tf.Variable) or isinstance(self._optimizer.lr, tf.Tensor):
            return self._optimizer.lr
        # LR SCHEDULE CASE
        else:
            return self._optimizer.lr(self.step)
        

    
    def __call__(self, img):
        """
        nuthin here yet
        """
        pass
            
        
    def __del__(self):
        if hasattr(self, "_mlflow"):
            import mlflow
            mlflow.end_run()