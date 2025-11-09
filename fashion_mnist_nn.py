import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class FashionMNISTClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def load_data(self):
        """
        Load and preprocess Fashion MNIST dataset
        """
        print("Loading Fashion MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape data to include channel dimension
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Split training data into training and validation sets
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Validation data shape: {x_val.shape}")
        print(f"Test data shape: {x_test.shape}")
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
    def build_model(self, hidden_layers=[128, 64], activation='relu', 
                   output_activation='softmax', dropout_rate=0.3):
        """
        Build neural network model
        """
        print("Building neural network model...")
        
        model = keras.Sequential()
        
        # Flatten input
        model.add(keras.layers.Flatten(input_shape=(28, 28, 1)))
        
        # Add hidden layers
        for neurons in hidden_layers:
            model.add(keras.layers.Dense(neurons, activation=activation))
            model.add(keras.layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(keras.layers.Dense(10, activation=output_activation))
        
        # Use Adam with AMSGrad and gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            amsgrad=True,
            clipnorm=1.0
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        self.model = model
        return model
    
    def train_model(self, x_train, y_train, x_val, y_val, epochs=20, batch_size=128):
        """
        Train the neural network model
        """
        print("Training model...")
        
        # Callbacks for early stopping and reducing learning rate
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, x_train, y_train, x_val, y_val, x_test, y_test):
        """
        Evaluate model on training, validation, and test sets
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Training evaluation
        train_loss, train_accuracy = self.model.evaluate(x_train, y_train, verbose=0)
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        # Validation evaluation
        val_loss, val_accuracy = self.model.evaluate(x_val, y_val, verbose=0)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Test evaluation
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return {
            'train': {'loss': train_loss, 'accuracy': train_accuracy},
            'val': {'loss': val_loss, 'accuracy': val_accuracy},
            'test': {'loss': test_loss, 'accuracy': test_accuracy}
        }
    
    def plot_training_history(self):
        """
        Plot training history (loss and accuracy curves)
        """
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, x_test, y_test):
        """
        Plot confusion matrix for test set predictions
        """
        print("Generating confusion matrix...")
        
        # Make predictions
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Fashion MNIST')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, 
                                  target_names=self.class_names))
        
        return cm
    
    def visualize_predictions(self, x_test, y_test, num_samples=10):
        """
        Visualize sample predictions
        """
        # Make predictions
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Select random samples
        indices = np.random.choice(len(x_test), num_samples, replace=False)
        
        # Plot samples
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            axes[i].imshow(x_test[idx].reshape(28, 28), cmap='gray')
            true_label = self.class_names[y_test[idx]]
            pred_label = self.class_names[y_pred_classes[idx]]
            confidence = np.max(y_pred[idx])
            
            color = 'green' if y_test[idx] == y_pred_classes[idx] else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', 
                            color=color, fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to run the Fashion MNIST classification
    """
    # Initialize classifier
    classifier = FashionMNISTClassifier()
    
    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = classifier.load_data()
    
    # Build model
    classifier.build_model(
        hidden_layers=[256, 128, 64],  # Three hidden layers
        activation='relu',
        dropout_rate=0.3
    )
    
    # Train model
    classifier.train_model(
        x_train, y_train, 
        x_val, y_val,
        epochs=20,
        batch_size=128
    )
    
    # Evaluate model
    results = classifier.evaluate_model(x_train, y_train, x_val, y_val, x_test, y_test)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(x_test, y_test)
    
    # Visualize sample predictions
    classifier.visualize_predictions(x_test, y_test)
    
    return classifier, results

if __name__ == "__main__":
    classifier, results = main()
