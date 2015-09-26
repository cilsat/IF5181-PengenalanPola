import tornado.httpserver
import tornado.ioloop
import tornado.web

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import io
import os.path
import random
import string

import util
 
def main():
    # Start server
    print('Starting server')
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(8080)
    tornado.ioloop.IOLoop.instance().start()

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler),
            (r"/test.png", ImageHandler),
            (r"/upload", UploadHandler)
        ]

        tornado.web.Application.__init__(self, handlers)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("upload_form.html")
        #self.write('<img src="test.png" />')
        
 
class ImageHandler(tornado.web.RequestHandler):
    def get(self):
        image = genImage(1.4)
        self.set_header('Content-type', 'image/png')
        self.set_header('Content-length', len(image))
        self.write(image)

class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        file1 = self.request.files['file1'][0]
        original_fname = file1['filename']
        extension = os.path.splitext(original_fname)[1]
        fname = ''.join(random.choice(string.ascii_lowercase + string.digits) for x in xrange(6))
        final_filename = fname+extension
        with open(final_filename, 'w') as out_file:
            out_file.write(file1['body'])

        font = "dejavusans-alphanumeric"
        fontimg = mpimg.imread('train/' + font + '.jpg')
        util.train(fontimg, font)

        testimg = mpimg.imread(final_filename)
        self.write(util.test(testimg, font))

        #self.finish("file" + final_filename + " is uploaded")

def genImage(freq):
    t = np.linspace(0, 10, 500)
    y = np.sin(t * freq * 2 * 3.141)
    fig1 = plt.figure()
    plt.plot(t, y)
    plt.xlabel('Time [s]')
    memdata = io.BytesIO()
    plt.grid(True)
    plt.savefig(memdata, format='png')
    image = memdata.getvalue()
    return image
 
if __name__ == "__main__":
    main()
