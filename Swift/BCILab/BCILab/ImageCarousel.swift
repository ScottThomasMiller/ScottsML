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
    
    convenience init() {
        self.init()
        self.animationImages = loadImages("Faces")
        self.contentMode = .center
        self.animationDuration = 3.0 * Double(self.animationImages!.count)
        self.startAnimating()
    }
}

struct ImageCarouselRep: UIViewRepresentable {
    func makeUIView(context: Context) -> ImageCarouselView {
        let newView = ImageCarouselView()
        return newView
    }
    
    func updateUIView(_ uiView: ImageCarouselView, context: Context) {
        print("ImageCarouseRep.updateUIView")
    }
    
    typealias UIViewType = ImageCarouselView
    
    
}
