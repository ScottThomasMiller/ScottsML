//
//  ImageCarousel.swift
//  BCILab
//
//  Created by Scott Miller on 7/24/21.
//

import Foundation
import SwiftUI

class ImageCarouselView: UIImageView {
    func loadImages(_ name: String) -> [UIImage] {
        var images = [UIImage]()
        let urls = Bundle.main.urls(forResourcesWithExtension: ".jpg", subdirectory: name)
        for url in urls! {
            guard let image = try? UIImage(data: Data(contentsOf: url)) else {
                print("Error loading image: \(url)")
                continue
            }
            images.append(image)
        }
        
        return images
    }
}

struct ImageCarouselRep: UIViewRepresentable {
    func makeUIView(context: Context) -> ImageCarouselView {
        let newView = ImageCarouselView()
        newView.animationImages = newView.loadImages("Faces")
        newView.contentMode = .center
        newView.animationDuration = 3.0 * Double(newView.animationImages!.count)
        newView.backgroundColor = .black
        newView.startAnimating()

        return newView
    }
    
    func updateUIView(_ uiView: ImageCarouselView, context: Context) {
        print("ImageCarouseRep.updateUIView")
    }
    
    typealias UIViewType = ImageCarouselView
    
    
}
