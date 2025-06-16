library(keras)
K <- keras::backend()
library(MLmetrics)
library(aricode)
library(clue)

### AUTOENCODER ###

autoencoder <- function(dims, 
                        activation = 'relu',
                        initializer = 'glorot_uniform') {
  
  n_layers <- length(dims) - 1
  # Input layer
  x <- layer_input(shape = dims[1], name = 'input')
  h <- x
  
  # Internal encoder layers
  for (i in seq_len(n_layers - 1)) {
    h <- h %>%
      layer_dense(units = dims[i + 1],
                  activation = activation,
                  kernel_initializer = initializer,
                  name = paste0('encoder_dense_', i))
  }
  
  # Bottleneck layer (embedding)
  h <- h %>%
    layer_dense(units = tail(dims, 1),
                activation = NULL,  # Explicitly no activation
                kernel_initializer = initializer,
                name = 'encoder')
  
  y <- h
  
  # Internal decoder layers
  for (i in seq(from = n_layers, to = 2, by = -1)) {
    y <- y %>%
      layer_dense(units = dims[i],
                  activation = activation,
                  kernel_initializer = initializer,
                  name = paste0('decoder_dense_', i - 1))
  }
  
  # Output layer
  y <- y %>%
    layer_dense(units = dims[1],
                kernel_initializer = initializer,
                name = 'decoder')
  
  return(list(
    autoencoderModel = keras_model(inputs = x, outputs = y),
    encoderModel = keras_model(inputs = x, outputs = h)
  ))
}



### CLUSTERING LAYER ###

ClusteringLayer <- R6::R6Class("ClusteringLayer",
                               inherit = KerasLayer,
                               lock_objects = FALSE,
                               public = list(
                                 n_clusters = 10,
                                 initial_weights = NULL,
                                 alpha = 1.0,
                                 name = '',
                                 
                                 initialize = function( n_clusters,
                                                        initial_weights = NULL, alpha = 1.0, name = '' ){
                                   
                                   self$n_clusters <- n_clusters
                                   self$initial_weights <- initial_weights
                                   self$alpha <- alpha
                                   self$name <- name
                                 },
                                 
                                 build = function(input_shape){
                                   
                                   if(length(input_shape)!= 2){
                                     stop(paste0("input_shape is not of length 2."))
                                   }
                                   
                                   self$clusters <- self$add_weight(
                                     shape = list(self$n_clusters, input_shape[[2]]),
                                     initializer = 'glorot_uniform', name = 'clusters' )
                                   
                                   if(!is.null(self$initial_weights )){
                                     self$set_weights(self$initial_weights)
                                     self$initial_weights <- NULL
                                   }
                                   self$built <- TRUE
                                 },
                                 
                                 call = function(inputs, mask = NULL) {
                                   diff <- K$expand_dims(inputs, axis = 1L) - self$clusters
                                   q <- 1.0 / (1.0 + (K$sum(K$square(diff), axis = 2L) / self$alpha))
                                   q <- K$pow(q, (self$alpha + 1.0) / 2.0)
                                   q <- q / K$sum(q, axis = 1L, keepdims = TRUE)
                                   return(q)
                                 }
                                 ,
                                 
                                 compute_output_shape = function( input_shape ){
                                   return(list(input_shape[[1]], self$n_clusters ))
                                 }
                               )
)

layer_clustering <- function(object,
                             n_clusters, initial_weights = NULL,
                             alpha = 1.0, name = '' )
{
  create_layer(ClusteringLayer, object,
               list( n_clusters = n_clusters,
                     initial_weights = initial_weights,
                     alpha = alpha, name = name )
  )
}

DEC <- R6::R6Class( "DEC", 
                    inherit = NULL, 
                    lock_objects = FALSE,
                    public = list(dims = NULL, 
                                  n_clusters = 10, 
                                  alpha = 1.0, 
                                  initializer = 'glorot_uniform', 
                                  initialize = function( dims,
                                                         n_clusters, alpha = 1.0, initializer = 'glorot_uniform',
                                                         convolutional = FALSE, inputImageSize = NULL ){
                                    self$dims <- dims
                                    self$n_clusters <- n_clusters
                                    self$alpha <- alpha
                                    self$initializer <- initializer
                                    self$convolutional <- convolutional
                                    self$inputImageSize <- inputImageSize
                                    
                                    ae <- autoencoder(self$dims,
                                                      initializer = self$initializer)
                                    self$autoencoder <- ae$autoencoderModel
                                    self$encoder <- ae$encoderModel
                                    
                                    
                                    # prepare DEC model
                                    
                                    clusteringLayer <- self$encoder$output %>%
                                      layer_clustering( self$n_clusters, name = "clustering" )
                                    
                                    self$model <- keras_model(inputs = self$encoder$input, 
                                                              outputs = clusteringLayer)
                                  },
                                  pretrain = function(x, optimizer = 'adam', epochs = 300L, batchSize = 256L, y = NULL) {
                                    
                                    interval <- max(as.integer(epochs /10))
                                    
                                    self$autoencoder$compile(optimizer = optimizer, loss = 'mse')
                                    
                                    for (epoch in seq_len(epochs)) {
                                      self$autoencoder$fit(x, x, batch_size = batchSize, epochs = 1L, verbose = 1)
                                      
                                      if (!is.null(y) && (epoch %% interval == 0)) {
                                        features <- self$encoder$predict(x, verbose = 0)
                                        km <- kmeans(features, centers = length(unique(y)), nstart = 20)
                                        acc <- ACC(y, km$cluster)
                                        nmi <- NMI(y, km$cluster)
                                        cat(sprintf("        |==>  acc: %.4f,  nmi: %.4f  <==|\n", acc, nmi))
                                      }
                                    }
                                    self$autoencoder$save_weights("autoencoder_pretrained.weights.h5")
                                    self$encoder$save_weights("encoder_pretrained.weights.h5")
                                    self$model$save_weights("dec_model_pretrained.weights.h5")
                                  },
                                  
                                  loadWeights = function(weights){
                                    
                                    self$model$load_weights(weights)
                                  },
                                  
                                  extractFeatures = function(x){
                                    
                                    self$encoder$predict(x, verbose = 0 )
                                  },
                                  
                                  predictClusterLabels = function(x){
                                    
                                    clusterProbabilities <- self$model$predict(x, verbose = 0)
                                    return(max.col(clusterProbabilities))
                                  },
                                  
                                  targetDistribution = function(q){
                                    
                                    weight <- q^2 / matrix(colSums(q), nrow = nrow(q), ncol = ncol(q), byrow = TRUE)
                                    p <- weight / matrix(rowSums(weight), nrow = nrow(weight), ncol = ncol(weight))
                                    
                                    return(p)
                                  },
                                  
                                  compile = function(optimizer = 'sgd', loss = 'kld', lossWeights = NULL, learning_rate = 1 ){
                                    
                                    self$model$compile(optimizer = optimizer, loss = loss, 
                                                       loss_weights = lossWeights )
                                  },
                                  
                                  fit = function(x, y = NULL, maxIter = 2e4, batchSize = 256L, 
                                                 tolerance = 1e-3, updateInterval = 500){
                                    
                                    km <- kmeans(self$encoder$predict(x), centers = 10, nstart = 20)
                                    previousPrediction <- km$cluster
                                    
                                    self$model$get_layer(name = 'clustering')$set_weights(list(km$centers))
                                    
                                    q <- self$model$predict(x, verbose = 0)
                                    p <- self$targetDistribution(q)
                                    
                                    loss <- 0
                                    index <- 0
                                    indexArray <- 1:nrow(x)
                                    
                                    for (i in seq_len(maxIter)) {
                                      if (i %% updateInterval == 1) {
                                        q <- self$model$predict(x, verbose = 0)
                                        p <- self$targetDistribution(q)
                                        currentPrediction <- max.col(q)
                                        deltaLabel <- sum(currentPrediction != previousPrediction) / length(currentPrediction)
                                        
                                        
                                        if (!is.null(y)) {
                                          cat(sprintf("Itr %d: deltaLabel = %.5f, ACC = %.5f, NMI = %.5f\n",
                                                      i, deltaLabel, ACC(y, currentPrediction), NMI(y, currentPrediction)))
                                        } else {
                                          cat(sprintf("Itr %d: deltaLabel = %.5f\n", i, deltaLabel))
                                          
                                        }
                                        
                                        previousPrediction <- currentPrediction
                                        
                                        if (i > 1 && deltaLabel < tolerance) {
                                          message("Reached tolerance threshold. Stopping training......")
                                          break
                                        }
                                      }
                                      if (i %% 50 == 0) {
                                        cat(sprintf("Itr %d: loss = %.5f\n", i, loss))
                                      }
                                      
                                      if (!is.null(p)) {
                                        batch_start <- index * batchSize + 1
                                        batch_end <- min((index + 1) * batchSize, nrow(x))
                                        
                                        if (batch_start > batch_end) {
                                          index <- 0
                                          next
                                        }
                                        
                                        batchIndices <- indexArray[batch_start:batch_end]
                                        loss <- self$model$train_on_batch(x = x[batchIndices, ], y = p[batchIndices, ])
                                        # cat(sprintf("Itr %d: loss = %.5f\n", i, loss))
                                      }
                                      
                                      index <- if ((index + 1) * batchSize <= nrow(x)) index + 1 else 0
                                    }
                                    
                                    return(currentPrediction)
                                  }
                    )
)


### DEEP EMBEDDED CLUSTERING CLASS ###

IDEC <- R6::R6Class("IDEC",
                    inherit = DEC,
                    lock_objects = FALSE,
                    public = list(
                      gamma = 0.1,
                      
                      initialize = function(dims, n_clusters, alpha = 1.0, gamma = 0.1, initializer = 'glorot_uniform') {
                        super$initialize(dims = dims, n_clusters = n_clusters, alpha = alpha, initializer = initializer)
                        self$gamma <- gamma
                        
                        # rebuild IDEC model with two outputs: clustering and decoder
                        clustering_output <- self$model$output
                        decoder_output <- self$autoencoder$output
                        self$model <- keras_model(inputs = self$encoder$input, outputs = list(clustering_output, decoder_output))
                      },
                      
                      compile = function(optimizer = 'sgd', loss_weights = NULL) {
                        loss_weights <- loss_weights %||% list(1, self$gamma)
                        self$model$compile(
                          optimizer = optimizer,
                          loss = list('kld', 'mse'),
                          loss_weights = loss_weights
                        )
                      },
                      
                      load_encoder_weights = function(weights_path) {
                        self$autoencoder$load_weights(weights_path)
                        cat("Autoencoder weights loaded into IDEC model.\n")
                      },
                      
                      fit = function(x, y = NULL, maxIter = 2e4, batchSize = 256L, 
                                     tolerance = 1e-3, updateInterval = 500, saveDir = NULL) {
                        
                        km <- kmeans(self$encoder$predict(x), centers = self$n_clusters, nstart = 20)
                        previousPrediction <- km$cluster
                        self$model$get_layer(name = 'clustering')$set_weights(list(km$centers))
                        
                        q <- self$model$predict(x, verbose = 0)[[1]]
                        p <- self$targetDistribution(q)
                        index <- 0
                        indexArray <- 1:nrow(x)
                        loss <- 0
                        
                        for (i in seq_len(maxIter)) {
                          if (i %% updateInterval == 1) {
                            q <- self$model$predict(x, verbose = 0)[[1]]
                            p <- self$targetDistribution(q)
                            currentPrediction <- max.col(q)
                            deltaLabel <- sum(currentPrediction != previousPrediction) / length(currentPrediction)
                            
                            if (!is.null(y)) {
                              cat(sprintf("Itr %d: deltaLabel = %.5f, ACC = %.5f, NMI = %.5f\n",
                                          i, deltaLabel, ACC(y, currentPrediction), NMI(y, currentPrediction)))
                            } else {
                              cat(sprintf("Itr %d: deltaLabel = %.5f\n", i, deltaLabel))
                            }
                            
                            if (i > 1 && deltaLabel < tolerance) {
                              message("Reached tolerance threshold. Stopping training...")
                              break
                            }
                            
                            previousPrediction <- currentPrediction
                          }
                          
                          if (i %% 50 == 0) {
                            cat(sprintf("Itr %d: loss = %.5f\n", i, loss[[1]]))
                          }
                          
                          batch_start <- index * batchSize + 1
                          batch_end <- min((index + 1) * batchSize, nrow(x))
                          
                          if (batch_start > batch_end) {
                            index <- 0
                            next
                          }
                          
                          batchIndices <- indexArray[batch_start:batch_end]
                          loss <- self$model$train_on_batch(
                            x = x[batchIndices, ], 
                            y = list(p[batchIndices, ], x[batchIndices, ])
                          )
                          
                          index <- if ((index + 1) * batchSize <= nrow(x)) index + 1 else 0
                        }
                        
                        return(currentPrediction)
                      },
                      
                      save_autoencoder_weights = function(filepath = "autoencoder_pretrained.weights.h5") {
                        self$autoencoder$save_weights(filepath)
                        cat(sprintf("Autoencoder weights saved to %s\n", filepath))
                      }
                    )
)

### METRICS ###
ACC <- function(true_labels, pred_labels) {
  tab <- table(true_labels, pred_labels)
  sum(tab[cbind(1:nrow(tab), solve_LSAP(tab, maximum = TRUE))]) / length(true_labels)
}


################################################################################

# Loading Dataset   ------------------------------------------------------------
library(abind)
load_mnist <- function() {
  mnist <- dataset_mnist()
  x <- abind::abind(mnist$train$x, mnist$test$x, along = 1)  # ahora (70000,28,28)
  y <- c(mnist$train$y, mnist$test$y)
  x <- array_reshape(x, c(dim(x)[1], 28*28)) / 255
  cat('MNIST samples', dim(x), '\n')
  return(list(x = x, y = y))
}

# (1) Cargar los datos
mnist_data <- load_mnist()
x <- mnist_data$x
y <- mnist_data$y
n_clusters <- length(unique(y))

# (2) Crear el modelo IDEC con la misma arquitectura que DEC
idec <- IDEC$new(dims = c(784, 500, 500, 2000, 10), n_clusters = n_clusters)

# (3) Cargar SOLO los pesos del autoencoder
idec$load_encoder_weights("autoencoder_pretrained.weights.h5")

# (4) Compilar el modelo con las dos pérdidas que usa IDEC
idec$compile(optimizer = optimizer_sgd(learning_rate = 0.01, momentum = 0.9))

# (5) Entrenar el modelo IDEC
y_pred <- idec$fit(x, y = y)
