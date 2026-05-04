/*
 * ***** ENTIRE HEADER FILE WRITTEN BY CLAUDE *****
 * 
 * imshow.h  –  Cross-platform image display for C projects
 *
 * Drop this single header into your project and call:
 *
 *   imshow("Window Title", pixels, width, height, channels);
 *   imshow_wait();        // block until ALL open windows are closed
 *
 * Supported channel layouts
 *   channels == 1  : Grayscale  (one byte per pixel)
 *   channels == 3  : RGB        (R, G, B bytes per pixel)
 *   channels == 4  : RGBA       (R, G, B, A bytes per pixel – alpha ignored on display)
 *
 * Pixel array format
 *   int pixels[height * width * channels]  with values 0-255.
 *   Row-major, top-to-bottom, left-to-right (same as OpenCV / stb_image).
 *
 * Platform notes
 *   Windows : uses Win32 GDI – no extra libraries, no linker flags needed.
 *             In Visual Studio the default project already links user32.lib
 *             and gdi32.lib, so everything compiles out of the box.
 *   Linux   : uses Xlib. Add  -lX11  to your linker flags (or in VS Code /
 *             CMake: target_link_libraries(myapp X11)).
 *             Xlib is installed by default on virtually every Linux desktop.
 *             If it is missing:  sudo apt install libx11-dev   (Debian/Ubuntu)
 *                                sudo dnf install libX11-devel (Fedora/RHEL)
 *
 * Thread safety: Not thread-safe. Call from the main thread only.
 *
 * License: MIT – use freely in any project.
 */

#ifndef IMSHOW_H
#define IMSHOW_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

    /* =========================================================================
     * Public API
     * ========================================================================= */

     /*
      * imshow() – Open a window and display the image.
      *
      *   title    : Window title string (UTF-8).
      *   pixels   : Pixel data array in planar CHW format, length = width * height * channels.
      *              Values must be in the range [0, 255].
      *   width    : Image width  in pixels (> 0).
      *   height   : Image height in pixels (> 0).
      *   channels : 1 (gray), 3 (RGB), or 4 (RGBA).
      *
      * The function returns immediately after the window is shown.
      * Call imshow_wait() to block until all windows are closed by the user.
      *
      * Returns 0 on success, -1 on error.
      */
    int imshow(const char* title, const int* pixels, int width, int height, int channels);

    /*
     * imshow_wait() – Process window events and block until all imshow windows
     *                 opened by this process have been closed by the user.
     *                 (Equivalent to cv2.waitKey(0) on the last window.)
     */
    void imshow_wait(void);


    /* =========================================================================
     * Internal helpers (shared between platforms)
     * ========================================================================= */

     /*
      * imshow__to_bgr() – Convert the caller's pixel array to a packed BGR byte
      * buffer suitable for display (Win32 DIB wants BGR; X11 wants 0x00RRGGBB in
      * a uint32 but we handle that separately). The returned buffer must be freed
      * by the caller.
      *
      * For X11 the caller should use imshow__to_xrgb() instead.
      */
    static unsigned char* imshow__to_bgr(const int* pixels, int width, int height, int channels)
    {
        int npix = width * height;
        unsigned char* buf = (unsigned char*)malloc((size_t)npix * 3);
        if (!buf) return NULL;

        /* Planar CHW layout: pixels[channel * height * width + row * width + col]
         * i.e. all R values come first, then all G values, then all B values.   */
        for (int i = 0; i < npix; i++) {
            int r, g, b;
            if (channels == 1) {
                int v = pixels[i];
                if (v < 0) v = 0; if (v > 255) v = 255;
                r = g = b = v;
            }
            else {
                /* channels == 3 or 4: plane 0 = R, plane 1 = G, plane 2 = B */
                r = pixels[0 * npix + i]; if (r < 0) r = 0; if (r > 255) r = 255;
                g = pixels[1 * npix + i]; if (g < 0) g = 0; if (g > 255) g = 255;
                b = pixels[2 * npix + i]; if (b < 0) b = 0; if (b > 255) b = 255;
            }
            buf[i * 3 + 0] = (unsigned char)b;
            buf[i * 3 + 1] = (unsigned char)g;
            buf[i * 3 + 2] = (unsigned char)r;
        }
        return buf;
    }

    /* =========================================================================
     * WINDOWS IMPLEMENTATION
     * ========================================================================= */
#if defined(_WIN32) || defined(_WIN64)

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

     /* Maximum number of imshow windows open simultaneously */
#define IMSHOW__MAX_WINDOWS 64

    typedef struct {
        HWND    hwnd;
        int     width;
        int     height;
        HBITMAP hbmp;        /* owned by us, must be DeleteObject'd */
        HDC     hmemdc;      /* compatible DC holding hbmp          */
        char    title[256];  /* window title, used to find existing windows */
    } imshow__WinState;

    /* We store per-window state in a flat array (simple, no heap bookkeeping) */
    static imshow__WinState  imshow__wins[IMSHOW__MAX_WINDOWS];
    static int               imshow__win_count = 0;
    static ATOM              imshow__wc_atom = 0;
    static HINSTANCE         imshow__hinst = NULL;

    static imshow__WinState* imshow__find_win(HWND hwnd)
    {
        for (int i = 0; i < imshow__win_count; i++)
            if (imshow__wins[i].hwnd == hwnd) return &imshow__wins[i];
        return NULL;
    }

    static imshow__WinState* imshow__find_win_by_title(const char* title)
    {
        for (int i = 0; i < imshow__win_count; i++)
            if (strcmp(imshow__wins[i].title, title) == 0) return &imshow__wins[i];
        return NULL;
    }

    static void imshow__remove_win(HWND hwnd)
    {
        for (int i = 0; i < imshow__win_count; i++) {
            if (imshow__wins[i].hwnd == hwnd) {
                /* Free GDI objects */
                if (imshow__wins[i].hmemdc)  DeleteDC(imshow__wins[i].hmemdc);
                if (imshow__wins[i].hbmp)    DeleteObject(imshow__wins[i].hbmp);
                /* Compact array */
                memmove(&imshow__wins[i], &imshow__wins[i + 1],
                    (size_t)(imshow__win_count - i - 1) * sizeof(imshow__WinState));
                imshow__win_count--;
                return;
            }
        }
    }

    static LRESULT CALLBACK imshow__WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
    {
        switch (msg) {
        case WM_PAINT: {
            imshow__WinState* s = imshow__find_win(hwnd);
            if (s && s->hmemdc) {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hwnd, &ps);
                BitBlt(hdc, 0, 0, s->width, s->height, s->hmemdc, 0, 0, SRCCOPY);
                EndPaint(hwnd, &ps);
            }
            return 0;
        }
        case WM_KEYDOWN:
            /* Any key press closes this window (like OpenCV) */
            DestroyWindow(hwnd);
            return 0;
        case WM_DESTROY:
            imshow__remove_win(hwnd);
            return 0;
        }
        return DefWindowProcA(hwnd, msg, wp, lp);
    }

    static int imshow__register_class(void)
    {
        if (imshow__wc_atom) return 1; /* already registered */

        imshow__hinst = GetModuleHandleA(NULL);

        WNDCLASSEXA wc;
        memset(&wc, 0, sizeof(wc));
        wc.cbSize = sizeof(wc);
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = imshow__WndProc;
        wc.hInstance = imshow__hinst;
        wc.hCursor = LoadCursorA(NULL, (LPCSTR)IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        wc.lpszClassName = "imshow_h_class";

        imshow__wc_atom = RegisterClassExA(&wc);
        return imshow__wc_atom != 0;
    }

    int imshow(const char* title, const int* pixels, int width, int height, int channels)
    {
        if (!pixels || width <= 0 || height <= 0 ||
            (channels != 1 && channels != 3 && channels != 4)) {
            fprintf(stderr, "imshow: invalid arguments\n");
            return -1;
        }
        if (!imshow__register_class()) {
            fprintf(stderr, "imshow: RegisterClassEx failed (%lu)\n", GetLastError());
            return -1;
        }

        const char* safe_title = title ? title : "imshow";

        /* ---- Convert pixels -> packed BGR bytes ---- */
        unsigned char* bgr = imshow__to_bgr(pixels, width, height, channels);
        if (!bgr) { fprintf(stderr, "imshow: out of memory\n"); return -1; }

        /* ---- Build new DIB section ---- */
        BITMAPINFOHEADER bih;
        memset(&bih, 0, sizeof(bih));
        bih.biSize = sizeof(bih);
        bih.biWidth = width;
        bih.biHeight = -height; /* negative = top-down */
        bih.biPlanes = 1;
        bih.biBitCount = 24;
        bih.biCompression = BI_RGB;
        int stride = ((width * 3) + 3) & ~3;

        HDC hscreendc = GetDC(NULL);
        HDC hmemdc = CreateCompatibleDC(hscreendc);
        ReleaseDC(NULL, hscreendc);

        void* dibits = NULL;
        HBITMAP hbmp = CreateDIBSection(hmemdc, (BITMAPINFO*)&bih,
            DIB_RGB_COLORS, &dibits, NULL, 0);
        if (!hbmp || !dibits) {
            free(bgr);
            DeleteDC(hmemdc);
            fprintf(stderr, "imshow: CreateDIBSection failed\n");
            return -1;
        }
        SelectObject(hmemdc, hbmp);

        for (int row = 0; row < height; row++) {
            unsigned char* dst = (unsigned char*)dibits + (size_t)row * stride;
            unsigned char* src = bgr + (size_t)row * width * 3;
            memcpy(dst, src, (size_t)width * 3);
        }
        free(bgr);

        /* ---- Reuse existing window if one with the same title is open ---- */
        imshow__WinState* s = imshow__find_win_by_title(safe_title);
        if (s) {
            /* Free old GDI objects */
            DeleteDC(s->hmemdc);
            DeleteObject(s->hbmp);

            /* Install new bitmap */
            s->hmemdc = hmemdc;
            s->hbmp = hbmp;

            /* Resize window if image dimensions changed */
            if (s->width != width || s->height != height) {
                s->width = width;
                s->height = height;

                RECT rc = { 0, 0, width, height };
                DWORD style = (DWORD)GetWindowLongA(s->hwnd, GWL_STYLE);
                AdjustWindowRectEx(&rc, style, FALSE, 0);
                SetWindowPos(s->hwnd, NULL, 0, 0,
                    rc.right - rc.left, rc.bottom - rc.top,
                    SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
            }

            /* Force a repaint */
            InvalidateRect(s->hwnd, NULL, FALSE);
            UpdateWindow(s->hwnd);
            return 0;
        }

        /* ---- No existing window — create a new one ---- */
        if (imshow__win_count >= IMSHOW__MAX_WINDOWS) {
            fprintf(stderr, "imshow: too many open windows\n");
            DeleteDC(hmemdc);
            DeleteObject(hbmp);
            return -1;
        }

        RECT rc = { 0, 0, width, height };
        DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        DWORD exstyle = 0;
        AdjustWindowRectEx(&rc, style, FALSE, exstyle);

        int ww = rc.right - rc.left;
        int wh = rc.bottom - rc.top;

        int sw = GetSystemMetrics(SM_CXSCREEN);
        int sh = GetSystemMetrics(SM_CYSCREEN);
        int wx = (sw - ww) / 2; if (wx < 0) wx = 0;
        int wy = (sh - wh) / 2; if (wy < 0) wy = 0;

        HWND hwnd = CreateWindowExA(exstyle,
            "imshow_h_class",
            safe_title,
            style,
            wx, wy, ww, wh,
            NULL, NULL, imshow__hinst, NULL);
        if (!hwnd) {
            DeleteDC(hmemdc);
            DeleteObject(hbmp);
            fprintf(stderr, "imshow: CreateWindowEx failed (%lu)\n", GetLastError());
            return -1;
        }

        s = &imshow__wins[imshow__win_count++];
        s->hwnd = hwnd;
        s->width = width;
        s->height = height;
        s->hbmp = hbmp;
        s->hmemdc = hmemdc;
        strncpy(s->title, safe_title, sizeof(s->title) - 1);
        s->title[sizeof(s->title) - 1] = '\0';

        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);
        return 0;
    }

    void imshow_wait(void)
    {
        MSG msg;
        while (imshow__win_count > 0) {
            /* GetMessage blocks until a message is available */
            BOOL ret = GetMessageA(&msg, NULL, 0, 0);
            if (ret == 0 || ret == -1) break; /* WM_QUIT or error */
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
    }

    /* =========================================================================
     * LINUX (X11) IMPLEMENTATION
     * ========================================================================= */
#elif defined(__linux__) || defined(__unix__)

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#define IMSHOW__MAX_WINDOWS 64

    typedef struct {
        Window   win;
        XImage* ximg;
        char* ximg_data; /* pixel buffer owned by us */
        int      width;
        int      height;
        char     title[256]; /* window title, used to find existing windows */
    } imshow__WinState;

    static Display* imshow__dpy = NULL;
    static imshow__WinState  imshow__wins[IMSHOW__MAX_WINDOWS];
    static int               imshow__win_count = 0;
    static Atom              imshow__wm_delete = 0; /* WM_DELETE_WINDOW */

    static int imshow__ensure_display(void)
    {
        if (imshow__dpy) return 1;
        imshow__dpy = XOpenDisplay(NULL);
        if (!imshow__dpy) {
            fprintf(stderr, "imshow: Cannot open X display. Is DISPLAY set?\n");
            return 0;
        }
        imshow__wm_delete = XInternAtom(imshow__dpy, "WM_DELETE_WINDOW", False);
        return 1;
    }

    static imshow__WinState* imshow__find_win(Window w)
    {
        for (int i = 0; i < imshow__win_count; i++)
            if (imshow__wins[i].win == w) return &imshow__wins[i];
        return NULL;
    }

    static imshow__WinState* imshow__find_win_by_title(const char* title)
    {
        for (int i = 0; i < imshow__win_count; i++)
            if (strcmp(imshow__wins[i].title, title) == 0) return &imshow__wins[i];
        return NULL;
    }

    static void imshow__remove_win(Window w)
    {
        for (int i = 0; i < imshow__win_count; i++) {
            if (imshow__wins[i].win == w) {
                /* XDestroyImage frees ximg_data for us when data was not
                 * malloc'd by Xlib, but since we set ->data ourselves we
                 * must free it and NULL the pointer before calling
                 * XDestroyImage to avoid a double-free.          */
                if (imshow__wins[i].ximg) {
                    imshow__wins[i].ximg->data = NULL; /* prevent Xlib free */
                    XDestroyImage(imshow__wins[i].ximg);
                }
                free(imshow__wins[i].ximg_data);
                XDestroyWindow(imshow__dpy, w);
                memmove(&imshow__wins[i], &imshow__wins[i + 1],
                    (size_t)(imshow__win_count - i - 1) * sizeof(imshow__WinState));
                imshow__win_count--;
                return;
            }
        }
    }

    int imshow(const char* title, const int* pixels, int width, int height, int channels)
    {
        if (!pixels || width <= 0 || height <= 0 ||
            (channels != 1 && channels != 3 && channels != 4)) {
            fprintf(stderr, "imshow: invalid arguments\n");
            return -1;
        }
        if (!imshow__ensure_display()) return -1;

        const char* safe_title = title ? title : "imshow";
        int screen = DefaultScreen(imshow__dpy);
        Visual* vis = DefaultVisual(imshow__dpy, screen);
        int depth = DefaultDepth(imshow__dpy, screen);

        /* ---- Build a 32-bit XRGB pixel buffer ---- */
        int npix = width * height;
        char* imgdata = (char*)malloc((size_t)npix * 4);
        if (!imgdata) { fprintf(stderr, "imshow: out of memory\n"); return -1; }

        /* Planar CHW layout: pixels[channel * height * width + row * width + col]
         * i.e. all R values come first, then all G values, then all B values.   */
        unsigned int* out = (unsigned int*)imgdata;
        for (int i = 0; i < npix; i++) {
            unsigned int r, g, b;
            if (channels == 1) {
                unsigned int v = (unsigned int)pixels[i];
                if (v > 255) v = 255;
                r = g = b = v;
            }
            else {
                /* plane 0 = R, plane 1 = G, plane 2 = B */
                r = (unsigned int)pixels[0 * npix + i]; if (r > 255) r = 255;
                g = (unsigned int)pixels[1 * npix + i]; if (g > 255) g = 255;
                b = (unsigned int)pixels[2 * npix + i]; if (b > 255) b = 255;
            }
            out[i] = (r << 16) | (g << 8) | b;
        }

        /* ---- Create XImage from new buffer ---- */
        XImage* ximg = XCreateImage(imshow__dpy, vis, (unsigned int)depth,
            ZPixmap, 0, imgdata,
            (unsigned int)width, (unsigned int)height,
            32, 0);
        if (!ximg) {
            free(imgdata);
            fprintf(stderr, "imshow: XCreateImage failed\n");
            return -1;
        }
        ximg->data = imgdata;

        /* ---- Reuse existing window if one with the same title is open ---- */
        imshow__WinState* s = imshow__find_win_by_title(safe_title);
        if (s) {
            /* Free old XImage (null out data pointer first to prevent double-free) */
            s->ximg->data = NULL;
            XDestroyImage(s->ximg);
            free(s->ximg_data);

            s->ximg = ximg;
            s->ximg_data = imgdata;

            /* Resize window if dimensions changed */
            if (s->width != width || s->height != height) {
                s->width = width;
                s->height = height;

                /* Update fixed-size hints */
                XSizeHints* hints = XAllocSizeHints();
                if (hints) {
                    hints->flags = PMinSize | PMaxSize;
                    hints->min_width = hints->max_width = width;
                    hints->min_height = hints->max_height = height;
                    XSetWMNormalHints(imshow__dpy, s->win, hints);
                    XFree(hints);
                }
                XResizeWindow(imshow__dpy, s->win,
                    (unsigned int)width, (unsigned int)height);
            }

            /* Redraw immediately */
            GC gc = DefaultGC(imshow__dpy, screen);
            XPutImage(imshow__dpy, s->win, gc, ximg,
                0, 0, 0, 0, (unsigned int)width, (unsigned int)height);
            XFlush(imshow__dpy);
            return 0;
        }

        /* ---- No existing window — create a new one ---- */
        if (imshow__win_count >= IMSHOW__MAX_WINDOWS) {
            fprintf(stderr, "imshow: too many open windows\n");
            ximg->data = NULL;
            XDestroyImage(ximg);
            free(imgdata);
            return -1;
        }

        unsigned long black = BlackPixel(imshow__dpy, screen);
        unsigned long white = WhitePixel(imshow__dpy, screen);

        int sw = DisplayWidth(imshow__dpy, screen);
        int sh = DisplayHeight(imshow__dpy, screen);
        int wx = (sw - width) / 2; if (wx < 0) wx = 0;
        int wy = (sh - height) / 2; if (wy < 0) wy = 0;

        Window win = XCreateSimpleWindow(imshow__dpy,
            RootWindow(imshow__dpy, screen),
            wx, wy,
            (unsigned int)width, (unsigned int)height,
            0, black, white);

        XSizeHints* hints = XAllocSizeHints();
        if (hints) {
            hints->flags = PMinSize | PMaxSize;
            hints->min_width = hints->max_width = width;
            hints->min_height = hints->max_height = height;
            XSetWMNormalHints(imshow__dpy, win, hints);
            XFree(hints);
        }

        XStoreName(imshow__dpy, win, safe_title);
        XSelectInput(imshow__dpy, win, ExposureMask | KeyPressMask);
        XSetWMProtocols(imshow__dpy, win, &imshow__wm_delete, 1);

        s = &imshow__wins[imshow__win_count++];
        s->win = win;
        s->ximg = ximg;
        s->ximg_data = imgdata;
        s->width = width;
        s->height = height;
        strncpy(s->title, safe_title, sizeof(s->title) - 1);
        s->title[sizeof(s->title) - 1] = '\0';

        XMapWindow(imshow__dpy, win);
        XFlush(imshow__dpy);
        return 0;
    }

    void imshow_wait(void)
    {
        if (!imshow__dpy || imshow__win_count == 0) return;

        while (imshow__win_count > 0) {
            XEvent ev;
            XNextEvent(imshow__dpy, &ev);

            switch (ev.type) {
            case Expose: {
                if (ev.xexpose.count != 0) break; /* skip intermediate Expose events */
                imshow__WinState* s = imshow__find_win(ev.xexpose.window);
                if (!s) break;
                GC gc = DefaultGC(imshow__dpy, DefaultScreen(imshow__dpy));
                XPutImage(imshow__dpy, s->win, gc, s->ximg,
                    0, 0, 0, 0,
                    (unsigned int)s->width, (unsigned int)s->height);
                break;
            }
            case KeyPress:
                /* Any key closes this window */
                imshow__remove_win(ev.xkey.window);
                break;
            case ClientMessage:
                /* Window manager "X" close button */
                if ((Atom)ev.xclient.data.l[0] == imshow__wm_delete)
                    imshow__remove_win(ev.xclient.window);
                break;
            }
        }

        XCloseDisplay(imshow__dpy);
        imshow__dpy = NULL;
    }

#else
     /*
      * Unsupported platform stub — produces a clear compile-time error.
      */
#error "imshow.h: Unsupported platform. Only Windows and Linux/X11 are supported."
#endif /* platform */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IMSHOW_H */