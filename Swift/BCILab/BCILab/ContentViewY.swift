//
//  ContentView.swift
//  BCILab
//
//  Created by Scott Miller on 7/24/21.
//

import SwiftUI

struct ContentViewY: View {
    var body: some View {
        GeometryReader { geometry in
            VStack {
                ImageCarouselRep()
            }
        }
    }
}

struct ContentViewY_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
