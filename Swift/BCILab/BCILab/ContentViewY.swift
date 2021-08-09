//
//  ContentView.swift
//  BCILab
//
//  Created by Scott Miller on 7/24/21.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        GeometryReader { geometry in
            VStack {
                ImageCarouselRep()
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
