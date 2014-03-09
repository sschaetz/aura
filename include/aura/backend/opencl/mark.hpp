#ifndef AURA_BACKEND_OPENCL_MARK_HPP
#define AURA_BACKEND_OPENCL_MARK_HPP

#include <CL/cl.h>
#include <boost/move/move.hpp>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/device.hpp>

namespace aura
{
namespace backend_detail
{
namespace opencl
{


/// callback used to free event asynchronously
void CL_CALLBACK delete_event_callback__(cl_event event, 
		cl_int event_command_exec_status, void* event_ptr) 
{
	AURA_OPENCL_SAFE_CALL(clReleaseEvent(event));
	delete (cl_event*)event_ptr;
}

/**
 * mark class
 */
class mark 
{

private:
	BOOST_MOVABLE_BUT_NOT_COPYABLE(mark)

public:
	/**
	 * create empty mark
	 */
	inline explicit mark() : event_(nullptr)
	{
	}

	/**
	 * create mark 
	 *
	 * @param f feed to create mark in 
	 */
	inline explicit mark(feed & f) : event_(new cl_event)
	{
		// clEnqueueMarkerWithWaitList only exists from OpenCL 1.2
		// we assume here that CL_VERSION_1_2 is defined for 2.0 too
#ifdef CL_VERSION_1_2
		AURA_OPENCL_SAFE_CALL(
			clEnqueueMarkerWithWaitList(
				f.get_backend_stream(), 
				0, NULL, event_
			)
		);
#else
		f.insert_event(event_);
#endif // CL_VERSION_1_2
	}

	/**
	 * move constructor, move mark information here, invalidate other
	 *
	 * @param m mark to move here
	 */
	mark(BOOST_RV_REF(mark) m) : event_(m.event_)
	{
		m.event_ = nullptr;
	}

	/**
	 * move assignment, move mark information here, invalidate other
	 *
	 * @param m mark to move here
	 */
	mark& operator=(BOOST_RV_REF(mark) m)
	{
		finalize();
		event_ = m.event_;
		m.event_ = nullptr;
		return *this;
	}

	/**
	 * destroy mark 
	 */
	inline ~mark()
	{
		finalize();
	}

	
private:
	/// finalize object (called from dtor and move assign)
	void finalize()
	{
		if(nullptr != event_) {
			// check if completed
			cl_int result;
			AURA_OPENCL_SAFE_CALL(
				clGetEventInfo(
					*event_, 
					CL_EVENT_COMMAND_EXECUTION_STATUS,
					sizeof(result),
					&result,
					NULL
				)
			);
			if(CL_COMPLETE == result) {
				clReleaseEvent(*event_);
				delete event_;
			} else {
				// enqueue callback
				AURA_OPENCL_SAFE_CALL(
					clSetEventCallback(
						*event_,
						CL_COMPLETE,
						&delete_event_callback__,
						event_
					)
				);
			}
		}
	}

	/// pointer to event
	cl_event * event_;

friend void insert(feed & f, mark & m);
friend void wait_for(mark & m);

};

/// insert marker into feed
void insert(feed & f, mark & m) 
{
	m.finalize();
	m.event_ = new cl_event;

#ifdef CL_VERSION_1_2
	AURA_OPENCL_SAFE_CALL(
		clEnqueueMarkerWithWaitList(
			f.get_backend_stream(), 
			0, NULL, m.event_
		)
	);
#else
		f.insert_event(m.event_);
#endif // CL_VERSION_1_2
}


void wait_for(mark & m) 
{
	AURA_OPENCL_SAFE_CALL(clWaitForEvents(1, m.event_));
}

} // opencl
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_MARK_HPP

