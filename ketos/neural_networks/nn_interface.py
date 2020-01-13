import tensorflow as tf
from .losses import FScoreLoss
from .metrics import precision_recall_accuracy_f


class RecipeCompat():
    """ Makes a loss function, metric or optimizer compatible with the Ketos recipe format.


        The resulting object can be included in a ketos recipe and read by the NNInterface (or it's subclasses)

        Args:
            name: str
                The name to be used in the recipe
            func: constructor
                The loss function, metric or optimizer constructor function
            kwargs
                Any keyword arguments to be passed to the constructor (func)

        Returns:
             A RecipeCompat object


        Examples:
          >>> # Example Metric
          >>> p = tf.keras.metrics.Precision
          >>> dec_p = RecipeCompat("precision", p)

          >>> # Example Optimizer
          >>> opt = tf.keras.optimizers.Adam
          >>> dec_opt = RecipeCompat("adam", opt, learning_rate=0.001)

          >>> # Example Loss
          >>> loss = tf.keras.losses.BinaryCrossentropy
          >>> dec_loss = RecipeCompat('binary_crossentropy', loss, from_logits=True)
    
    """
    def __repr__(self):
        return "{0} ketos recipe".format(self.name)

    def __init__(self, name, func, **kwargs):
        self.name = name
        self.func = func(**kwargs)
        self.args = kwargs

    def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        return result



class NNInterface():
    """ General interface for neural network architectures in the ketos.neural_networks module.

        This class implements common methods for neural network models.

    Args:
        neural_network: 
            An instance of one of the architectures available in the neural_networks module.
        data_input: data_feeder
            An object containing or able to read the data.

    
    """

    


    valid_optimizers = {'Adam':tf.keras.optimizers.Adam}
    valid_losses = {'FScoreLoss':FScoreLoss}
    valid_metrics = {'CategoricalAccuracy':tf.keras.metrics.CategoricalAccuracy}


    @classmethod
    def to1hot(cls, class_label, n_classes=2):
        one_hot = np.zeros(2)
        one_hot[class_label]=1.0
        return one_hot
    
    @classmethod
    def transform_train_batch(cls,x,y):
        X = x.reshape(x.shape[0],x.shape[1], x.shape[2],1)
        Y = np.array([cls.to1hot(sp) for sp in y])
        return (X,Y)

    @classmethod
    def transform_input(cls,input):
        if input.ndim == 2:
            transformed_input = input.reshape(1,input.shape[0], input.shape[1],1)
        elif input.ndim == 3:
            transformed_input = input.reshape(input.shape[0],input.shape[1], input.shape[2],1)
        else:
            raise ValueError("Expected input to have 2 or 3 dimensions, got {}({}) instead".format(input.ndims, input.shape))

        return transformed_input

    @classmethod
    def transform_output(cls,output):
        max_class = np.argmax(output, axis=-1)
        if output.shape[0] == 1:
            max_class_conf = output[0][max_class]
            transformed_output = (max_class[0], max_class_conf[0])
        elif output.shape[0] > 1:
            max_class_conf = np.array([output[i][c] for i, c in enumerate(max_class)])

        transformed_output = (max_class, max_class_conf)
        
        return transformed_output


    @classmethod
    def optimizer_from_recipe(cls, optimizer):
        name = optimizer['name']
        kwargs = optimizer['parameters']

        if name not in cls.valid_optimizers.keys():
            raise ValueError("Invalid optimizer name '{}'".format(name))
        built_optimizer = RecipeCompat(name,cls.valid_optimizers[name],**kwargs)

        return built_optimizer

    @classmethod
    def optimizer_to_recipe(cls, optimizer):
        name = optimizer.name
        kwargs = optimizer.args

        if name not in cls.valid_optimizers.keys():
            raise ValueError("Invalid optimizer name '{}'".format(name))
        recipe_optimizer = {'name':name, 'parameters':kwargs}

        return recipe_optimizer

    @classmethod
    def loss_function_from_recipe(cls, loss_function):
        name = loss_function['name']
        kwargs = loss_function['parameters']

        if name not in cls.valid_losses.keys():
            raise ValueError("Invalid loss function name '{}'".format(name))
        built_loss = RecipeCompat(name, cls.valid_losses[name],**kwargs)

        return built_loss

    @classmethod
    def loss_function_to_recipe(cls, loss_function):
        name = loss_function.name
        kwargs = loss_function.args

        if name not in cls.valid_losses.keys():
            raise ValueError("Invalid loss function name '{}'".format(name))
        recipe_loss = {'name':name, 'parameters':kwargs}

        return recipe_loss


    @classmethod
    def metrics_from_recipe(cls, metrics):
        
        built_metrics = []
        for m in metrics:
            name = m['name']
            kwargs = m['parameters']
             
            if name not in cls.valid_metrics.keys():
                raise ValueError("Invalid metric name '{}'".format(m['name']))
            built_metrics.append(RecipeCompat(name, cls.valid_metrics[name], **kwargs))

        return built_metrics

    @classmethod
    def metrics_to_recipe(cls, metrics):
        # name = metrics.name
        # kwargs = metrics.parameters

        recipe_metrics = []
        for m in metrics: 
            if m.name not in cls.valid_metrics.keys():
                raise ValueError("Invalid metric name '{}'".format(m['name']))
            recipe_metrics.append({'name':m.name, 'parameters':m.args})

        return recipe_metrics



    @classmethod
    def read_recipe_file(cls, json_file):
        with open(json_file, 'r') as json_recipe:
            recipe_dict = json.load(json_recipe)

        optimizer = cls.optimizer_from_recipe(recipe_dict['optimizer'])
        loss_function = cls.loss_function_from_recipe(recipe_dict['loss_function'])
        metrics = cls.metrics_from_recipe(recipe_dict['metrics'])

        recipe_dict['optimizer'] = optimizer
        recipe_dict['loss_function'] = loss_function
        recipe_dict['metrics'] = metrics

        return recipe_dict

    @classmethod
    def write_recipe_file(cls, json_file, recipe):
        with open(json_file, 'w') as json_recipe:
            json.dump(recipe, json_recipe)


    @classmethod
    def load(cls, recipe, weights_path):
        instance = cls.build_from_recipe(recipe) 
        latest_checkpoint = tf.train.latest_checkpoint(weights_path)
        instance.model.load_weights(latest_checkpoint)

        return instance

    @classmethod
    def load_model(cls, model_file, new_model_folder, overwrite=True):
        #tmp_dir = "ketos_tmp_model_files"
        try:
            os.makedirs(new_model_folder)
        except FileExistsError:
            if overwrite == True:
                rmtree(new_model_folder)
                os.makedirs(new_model_folder)
            else:
                raise FileExistsError("Ketos needs a new folder for this model. Choose a folder name that does not exist or set 'overwrite' to True to replace the existing folder")

        with ZipFile(model_file, 'r') as zip:
            zip.extractall(path=new_model_folder)
        recipe = cls.read_recipe_file(os.path.join(new_model_folder,"recipe.json"))
        model_instance = cls.load(recipe,  os.path.join(new_model_folder, "checkpoints"))

        #rmtree(tmp_dir)

        return model_instance
    
    @classmethod
    def build_from_recipe(cls, recipe):
        block_list = recipe['block_list']
        n_classes = recipe['n_classes']
        initial_filters = recipe['initial_filters']
        optimizer = recipe['optimizer']
        loss_function = recipe['loss_function']
        metrics = recipe['metrics']

        instance = cls(block_list=block_list, n_classes=n_classes, initial_filters=initial_filters, optimizer=optimizer, loss_function=loss_function, metrics=metrics)

        return instance

    def __init__(self, optimizer, loss_function, metrics):
        
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics

        self.model=None
        self.compile_model()
        #self.metrics_names = self.model.metrics_names

        
        self.log_dir = None
        self.checkpoint_dir = None
        self.tensorboard_callback = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def write_recipe(self):
        recipe = {}
        recipe['optimizer'] = self.optimizer_to_recipe(self.optimizer)
        recipe['loss_function'] = self.loss_function_to_recipe(self.loss_function)
        recipe['metrics'] = self.metrics_to_recipe(self.metrics)

        return recipe



    def save_recipe(self, recipe_file):
        recipe = self.write_recipe()
        self.write_recipe_file(json_file=recipe_file, recipe=recipe)

    def save_model(self, model_file):
        recipe_path = os.path.join(self.checkpoint_dir, 'recipe.json')
        with ZipFile(model_file, 'w') as zip:
            
            latest = tf.train.latest_checkpoint(self.checkpoint_dir)
            checkpoints = glob(latest + '*')                                                                                                                 
            self.save_recipe(recipe_path)
            zip.write(recipe_path, "recipe.json")
            zip.write(os.path.join(self.checkpoint_dir, "checkpoint"), "checkpoints/checkpoint")
            for c in checkpoints:
                 zip.write(c, os.path.join("checkpoints", os.path.basename(c)))            

        os.remove(recipe_path)



    
    def compile_model(self):
        self.model.compile(optimizer=self.optimizer.func,
                            loss = self.loss_function,
                            metrics = self.metrics)
        self.metrics_names = self.model.metrics_names

    def set_train_generator(self, train_generator):
        self.train_generator = train_generator

    def set_val_generator(self, val_generator):
        self.val_generator = val_generator

    def set_test_generator(self, test_generator):
        self.test_generator = test_generator

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def set_tensorboard_callback(self):
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        tensorboard_callback.set_model(self.model)
        
    def print_metrics(self, metric_values):
        message  = [self.metrics_names[i] + ": {} ".format(metric_values[i]) for i in range(len(self.metrics_names))]
        #import pdb; pdb.set_trace()
        print(''.join(message))

    def name_logs(self, logs, prefix="train_"):
        named_logs = {}
        for l in zip(self.metrics_names, logs):
            named_logs[prefix+l[0]] = l[1]
        return named_logs

    def train_loop(self, n_epochs, verbose=True, validate=True, log_tensorboard=False):
        for epoch in range(n_epochs):
            #Reset the metric accumulators
            train_precision = 0    
            train_recall = 0
            train_f_score = 0
            train_accuracy = 0
            val_precision = 0    
            val_recall = 0
            val_f_score = 0
            val_accuracy = 0
            self.model.reset_metrics()
                
            for train_batch_id in range(self.train_generator.n_batches):
                train_X, train_Y = next(self.train_generator)  
                train_result = self.model.train_on_batch(train_X, train_Y)

                train_set_pred = self.model.predict(train_X)
                train_f_p_r = precision_recall_accuracy_f(y_true=train_Y, y_pred=train_set_pred, f_beta=0.5)

                train_f_score += train_f_p_r['f_score']
                train_precision += train_f_p_r['precision']
                train_recall += train_f_p_r['recall']
                train_accuracy += train_f_p_r['accuracy']

                if verbose == True:
                    print("train: ","Epoch:{} Batch:{}".format(epoch, train_batch_id))
                    print("loss:{:.3f} accuracy:{:.3f} precision:{:.3f} recall:{:.3f} f-score:{:.3f}".format(
                        1-train_f_p_r['f_score'], train_f_p_r['accuracy'], train_f_p_r['precision'], train_f_p_r['recall'], train_f_p_r['f_score']) 
                    )
                    print("")


                
                    #self.print_metrics(train_result)
            train_precision = train_precision / self.train_generator.n_batches
            train_recall = train_recall / self.train_generator.n_batches
            train_f_score = train_f_score / self.train_generator.n_batches
            train_accuracy = train_accuracy / self.train_generator.n_batches
            
            if verbose == True:
                    print("====================================================================================")
                    print("train: ","Epoch:{}".format(epoch))
                    print("loss:{:.3f} accuracy:{:.3f} precision:{:.3f} recall:{:.3f} f-score:{:.3f}".format(
                       1 - train_f_score, train_accuracy, train_precision, train_recall, train_f_score) 
                    )
                    

            if log_tensorboard == True:
                self.tensorboard_callback.on_epoch_end(epoch, name_logs(train_result, "train_"))                
            if validate == True:
                for val_batch_id in range(self.val_generator.n_batches):
                    val_X, val_Y = next(self.val_generator)
                    val_result = self.model.test_on_batch(val_X, val_Y, 
                                                # return accumulated metrics
                                                reset_metrics=False)
                    

                    val_set_pred = self.model.predict(val_X)
                    val_f_p_r = precision_recall_accuracy_f(y_true=val_Y, y_pred=val_set_pred, f_beta=0.5)

                    val_f_score += val_f_p_r['f_score']
                    val_precision += val_f_p_r['precision']
                    val_recall += val_f_p_r['recall']
                    val_accuracy += val_f_p_r['accuracy']


                
                    #self.print_metrics(val_result)
                val_precision = val_precision / self.val_generator.n_batches
                val_recall = val_recall / self.val_generator.n_batches
                val_f_score = val_f_score / self.val_generator.n_batches
                val_accuracy = val_accuracy / self.val_generator.n_batches
                
                if verbose == True:
                        print("\nval: ")
                        print("loss:{:.3f} accuracy:{:.3f} precision:{:.3f} recall:{:.3f} f-score:{:.3f}".format(
                            1 - val_f_score, val_accuracy, val_precision, val_recall, val_f_score) 
                        )
                        print("====================================================================================")

                        


                # if verbose == True:
                #     print("\nval: ")
                #     self.print_metrics(val_result)
                if log_tensorboard == True:
                    self.tensorboard_callback.on_epoch_end(epoch, name_logs(val_result, "val_"))  

            
            # if epoch % 5:
            #     checkpoint_name = "cp-{:04d}.ckpt".format(epoch)
            #     self.model.save_weights(os.path.join(self.checkpoint_dir, checkpoint_name))
        
        if log_tensorboard == True:
            self.tensorboard_callback.on_train_end(None)

        
    def run(self, input, return_raw_output=False):
        input = self.transform_input(input)
        output = self.model.predict(input)
        
        if not return_raw_output:
            return self.transform_output(output)
        else:
            return output

    
    def run_on_batch(self, input_batch, return_raw_output=False, transform_input=True):
        if transform_input == True:
            input_batch = self.transform_input(input_batch)
        output = self.model.predict(input_batch)
        
        if not return_raw_output:
            return self.transform_output(output)
        else:
            return output


            

    
    
    