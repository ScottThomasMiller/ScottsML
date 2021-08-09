//
//  LabeledImage.swift
//  BCILab
//
//  Created by Scott Miller on 8/7/21.
//

import Foundation
import SwiftUI

struct LabeledImage {
    let image: UIImage
    let label: String
}

func appendImages(_ name: String,  labeledImages: inout [LabeledImage])  {
    let urls = Bundle.main.urls(forResourcesWithExtension: ".jpg", subdirectory: name)
    for url in urls! {
        guard let image = try? UIImage(data: Data(contentsOf: url)) else {
            print("Error loading image: \(url)")
            continue
        }
        let labeledImage = LabeledImage(image: image, label: name)
        labeledImages.append(labeledImage)
    }
}

func prepareImages () -> [LabeledImage] {
    // replace the labeled animation images with a new set, which is a shuffling of the current animation images with blanks inserted between each image:
    guard let blankURL = Bundle.main.url(forResource: "green_crosshair", withExtension: ".png") else {
        print("Error: cannot load blank image")
        return [LabeledImage]()
    }
    let blankImage = try! UIImage(data: Data(contentsOf: blankURL))
    let blank = LabeledImage(image: blankImage!, label: "blank")
    var labeledImages = [LabeledImage]()
    appendImages("Faces", labeledImages: &labeledImages)
    appendImages("NonFaces", labeledImages: &labeledImages)
    let shuffledImages = labeledImages.shuffled()
    var finalImages = [LabeledImage]()
    for image in shuffledImages {
        finalImages.append(blank)
        finalImages.append(image)
    }
    return finalImages
}
