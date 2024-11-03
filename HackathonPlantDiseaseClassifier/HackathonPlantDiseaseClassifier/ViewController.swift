//
//  ViewController.swift
//  HackathonPlantDiseaseClassifier
//
//  Created by Rohan Kaman on 17/10/2024.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController {
    
    // MARK: - Outlets
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!
    
    // MARK: - Properties
    
    // Lazy properties allow initialization when first accessed
    lazy var mlModel: MLModel = {
        // Attempt to load the model file
        guard let modelURL = Bundle.main.url(forResource: "PlantDiseaseModel", withExtension: "mlmodelc") else {
            fatalError("Failed to find the model file.")
        }
        
        do {
            let model = try MLModel(contentsOf: modelURL)
            return model
        } catch {
            fatalError("Failed to load the ML model: \(error)")
        }
    }()
    
    lazy var visionModel: VNCoreMLModel = {
        do {
            let vnModel = try VNCoreMLModel(for: mlModel)
            return vnModel
        } catch {
            fatalError("Failed to create VNCoreMLModel: \(error)")
        }
    }()
    
    // MARK: - View Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Additional setup if needed
        resultLabel.text = "Classification Result"
    }
    
    // MARK: - Image Classification
    
    func classifyImage(_ image: UIImage) {
        // Fix image orientation
        let fixedImage = image.fixedOrientation()
        
        // Resize image
        let resizedImage = resizeImage(fixedImage, targetSize: CGSize(width: 224, height: 224))
        
        // Convert UIImage to CVPixelBuffer with normalization
        guard let resizedImage = resizeImage(fixedImage, targetSize: CGSize(width: 224, height: 224)),
              let pixelBuffer = resizedImage.toCVPixelBuffer(width: 224, height: 224) else {
            fatalError("Unable to convert UIImage to CVPixelBuffer.")
        }

        // Create a Vision request
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            if let results = request.results as? [VNClassificationObservation],
               let topResult = results.first {
                
                DispatchQueue.main.async {
                    if topResult.confidence < 0.5 {
                        self?.resultLabel.text = "Low confidence. Try another image."
                    } else {
                        let confidence = Int(topResult.confidence * 100)
                        self?.resultLabel.text = "\(topResult.identifier) (\(confidence)%)"
                    }
                }
            } else if let error = error {
                DispatchQueue.main.async {
                    self?.resultLabel.text = "Error: \(error.localizedDescription)"
                }
            }
        }
        
        // Create a handler for the pixel buffer
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        // Perform the request on a background thread
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async {
                    self.resultLabel.text = "Failed to classify image: \(error.localizedDescription)"
                }
            }
        }
    }

    
    // MARK: - Actions
    
    @IBAction func selectImageButtonTapped(_ sender: UIButton) {
        // Present the image picker
        let imagePicker = UIImagePickerController()
        imagePicker.delegate = self
        imagePicker.sourceType = .photoLibrary
        imagePicker.allowsEditing = false
        present(imagePicker, animated: true, completion: nil)
    }
}

// MARK: - UIImagePickerControllerDelegate & UINavigationControllerDelegate

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        // Dismiss the picker if the user canceled
        picker.dismiss(animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
        // Retrieve the selected image
        picker.dismiss(animated: true, completion: nil)
        
        if let uiImage = info[.originalImage] as? UIImage {
            imageView.image = uiImage
            classifyImage(uiImage)
        }
    }
}
// MARK: - UIImage Extensions

extension UIImage {
    func fixedOrientation() -> UIImage {
        if imageOrientation == .up {
            return self
        }
        
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        draw(in: CGRect(origin: .zero, size: size))
        
        let normalizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return normalizedImage ?? self
    }
}

// MARK: - Helper Functions

func resizeImage(_ image: UIImage, targetSize: CGSize) -> UIImage? {
    let renderer = UIGraphicsImageRenderer(size: targetSize)
    let resizedImage = renderer.image { _ in
        image.draw(in: CGRect(origin: .zero, size: targetSize))
    }
    return resizedImage
}

extension UIImage {
    func toCVPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attributes: [NSObject: AnyObject] = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attributes as CFDictionary, &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        let data = CVPixelBufferGetBaseAddress(buffer)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: data, width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            CVPixelBufferUnlockBaseAddress(buffer, [])
            return nil
        }
        
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        draw(in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(buffer, [])
        
        return buffer
    }
}

