import SwiftUI

// forked from: https://stackoverflow.com/questions/58896661/swiftui-create-image-slider-with-dots-as-indicators

struct ContentViewX: View {
    public let timer = Timer.publish(every: 3, on: .main, in: .common).autoconnect()
    @State private var selection = 0

    ///  images with these names are placed  in my assets
    let images: [LabeledImage] = appendImages("Faces")

    var body: some View {
        ZStack{
            Color.black
            TabView(selection : $selection){
                ForEach(0..<5){ i in
                    Image("\(images[i].image)")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }
            }.tabViewStyle(PageTabViewStyle())
            .indexViewStyle(PageIndexViewStyle(backgroundDisplayMode: .always))
            .onReceive(timer, perform: { _ in
                withAnimation{
                    print("selection is",selection)
                    selection = selection < 5 ? selection + 1 : 0
                }
            })
        }
    }
}
