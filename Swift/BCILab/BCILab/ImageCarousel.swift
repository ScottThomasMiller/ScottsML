//
//  ImageCarousel.swift
//  BCILab
//
//  Created by Scott Miller on 7/24/21.
//

import Foundation
import SwiftUI
 
class ImageCarouselView: UIImageView {
    func appendImages(_ name: String)  {
        var images = [UIImage]()
        let urls = Bundle.main.urls(forResourcesWithExtension: ".jpg", subdirectory: name)
        for url in urls! {
            guard let image = try? UIImage(data: Data(contentsOf: url)) else {
                print("Error loading image: \(url)")
                continue
            }
            images.append(image)
        }
        
        if self.animationImages == nil {
            self.animationImages = images }
        else {
            self.animationImages! += images
        }
        
    }
    
    func prepare () {
        // replace the animation images with a new set, which is a shuffling of the current
        // animation images with blanks inserted between each image:
        guard let blankURL = Bundle.main.url(forResource: "green_crosshair", withExtension: ".png") else {
            print("Error: cannot load blank image")
            return
        }
        let blankImage = try! UIImage(data: Data(contentsOf: blankURL))
        if let currentImages = self.animationImages {
            let shuffledImages = currentImages.shuffled()
            var finalImages = [UIImage]()
            for image in shuffledImages {
                if finalImages.count > 0 {
                    finalImages.append(blankImage!)
                }
                finalImages.append(image)
            }
            self.animationImages = finalImages
        }
    }
}

struct ImageCarouselRep: UIViewRepresentable {
    typealias UIViewType = ImageCarouselView

    func makeUIView(context: Context) -> ImageCarouselView {
        let newView = ImageCarouselView()
        newView.appendImages("Faces")
        newView.appendImages("NonFaces")
        newView.prepare()
        newView.contentMode = .center
        newView.animationDuration = 2.0 * Double(newView.animationImages!.count)
        newView.backgroundColor = .black
        newView.animationRepeatCount = 1
        newView.startAnimating()
 
        let libinfo = String(cString: lsl_library_info())
        print(libinfo)
        let chFormat = ChannelFormat(format: lsl_channel_format_t(3))
        let streamInfo = StreamInfo(name: "BCILab", format: chFormat, id: "BCILab")
        let outlet = Outlet(streamInfo: streamInfo)

        return newView
    }
    
    func updateUIView(_ uiView: ImageCarouselView, context: Context) {
        print("ImageCarouseRep.updateUIView")
    }
}
