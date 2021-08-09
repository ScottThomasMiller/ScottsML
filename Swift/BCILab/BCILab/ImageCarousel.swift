//
//  ImageCarousel.swift
//  BCILab
//
//  Created by Scott Miller on 7/24/21.
//

import Foundation
import SwiftUI
 

class ImageCarouselView: UIView {
    var labeledImages = [LabeledImage]()
    @State private var imageNum = 0
    
    func pushAnimation() {
//        let libinfo = String(cString: lsl_library_info())
//        print("LSL library: \(libinfo)")
        let chFormat = ChannelFormat(format: lsl_channel_format_t(3))
        let streamInfo = StreamInfo(name: "ImageLabel", format: chFormat, id: "imageType",
                                    sampleRate: 1.0)
        let outlet = Outlet(streamInfo: streamInfo)

        defer {
            outlet.close()
        }
        
        for labeledImage in self.labeledImages {
            let imageView = UIImageView(image: labeledImage.image)
            imageView.frame = self.frame
            self.addSubview(imageView)
            self.bringSubviewToFront(imageView)
            self.imageNum += 1
            
            print("push label: \(labeledImage.label)")
            do {
                try outlet.push(data: labeledImage.label)
            }
            catch {
                print("Cannot push to outlet. Error code: \(error)")
            }
            sleep(1)
        }
    }
    
    func appendImages(_ name: String)  {
        let urls = Bundle.main.urls(forResourcesWithExtension: ".jpg", subdirectory: name)
        for url in urls! {
            guard let image = try? UIImage(data: Data(contentsOf: url)) else {
                print("Error loading image: \(url)")
                continue
            }
            let labeledImage = LabeledImage(image: image, label: name)
            self.labeledImages.append(labeledImage)
        }
    }
    
    func prepare () {
        // replace the labeled animation images with a new set, which is a shuffling of the current animation images with blanks inserted between each image:
        guard let blankURL = Bundle.main.url(forResource: "green_crosshair", withExtension: ".png") else {
            print("Error: cannot load blank image")
            return
        }
        let blankImage = try! UIImage(data: Data(contentsOf: blankURL))
        let blank = LabeledImage(image: blankImage!, label: "blank")
        let shuffledImages = self.labeledImages.shuffled()
        var finalImages = [LabeledImage]()
        for image in shuffledImages {
            finalImages.append(blank)
            finalImages.append(image)
        }
        self.labeledImages = finalImages
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
        //newView.animationDuration = 2.0 * Double(newView.animationImages!.count)
        newView.backgroundColor = .black
        //newView.animationRepeatCount = 1
        //newView.startAnimating()
        newView.pushAnimation()

        return newView
    }
    
    func updateUIView(_ uiView: ImageCarouselView, context: Context) {
        print("ImageCarouseRep.updateUIView")
    }
}
